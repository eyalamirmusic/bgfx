/*
 * Copyright 2011-2026 Branimir Karadzic. All rights reserved.
 * License: https://github.com/bkaradzic/bgfx/blob/master/LICENSE
 */

#include "bgfx_p.h"

#if BGFX_CONFIG_RENDERER_EACP
#	include "renderer_eacp.h"
#	include "renderer.h"

#	include <eacp/GPU/Buffer/StreamingBuffers.h>
#	include <eacp/GPU/Frame/ComputePass.h>
#	include <eacp/GPU/Pipeline/ComputePipeline.h>
#	include <eacp/GPU/Shader/ShaderLibrary.h>
#	include <eacp/GPU/Texture/Sampler.h>
#	include <eacp/GPU/Texture/Texture.h>

#	include <bimg/bimg.h>

#	include <atomic>
#	include <optional>
#	include <string>
#	include <unordered_map>
#	include <vector>

// The eacp backend (plan.md section 4). Gap numbers (Gn) are plan.md section 5's.

namespace bgfx { namespace eacp
{
	static GPU::Frame* s_frame;
	static std::atomic<bool> s_hiDpi;

	GPU::Frame* currentFrame()
	{
		return s_frame;
	}

	/// Things the backend cannot do yet, each reported once.
	struct Unsupported
	{
		enum Enum
		{
			SamplerKind,
			SwapChain,          // G16
			FrameBufferColor,   // attachments eacp's pass refuses together
			FrameBufferLayer,   // layered rendering
			FrameBufferFormat,  // G4
			FrameBufferDepth,
			DepthUnbound,
			ScreenShotWindow,   // G34
			DebugTextTarget,
			RenderImage,        // G40
			VertexTexture,      // G17
			SamplerSlot,        // G17
			TextureLayers,
			TextureSrgb,        // G4
			UnknownFormat,      // G4
			ComputeWriteFormat,
			TextureUpdate,
			TextureRead,
			TextureExternal,
			ClearTexture,
			BorderColor,
			ComputeImageFormat,
			ComputeReadWrite,
			ComputeBind,
			BlitDepth,          // G36
			BlitLevel,
			BlitVolume,
			Indirect,
			Msaa,               // window is sample count 1
			Capture,            // G34
			OcclusionQuery,     // G10
			BlendFactor,        // G26
			DepthBias,          // G27
			Wireframe,          // G28
			VertexFormat,       // G7
			UniformBlockSize,   // G9
			Viewport,           // G30

			Count
		};
	};

	static void traceOnce(Unsupported::Enum _what, const char* _message)
	{
		static bool s_reported[Unsupported::Count] = {};

		if (!s_reported[_what])
		{
			s_reported[_what] = true;
			BX_TRACE("eacp: %s", _message);
		}

		BX_UNUSED(_message);
	}

	/// The clear colour of colour attachment _attachment: its own palette
	/// entry under BGFX_CLEAR_COLOR_USE_PALETTE, the one packed colour
	/// otherwise.
	static ::eacp::Graphics::Color clearColor(const Clear& _clear, const float _palette[][4], uint32_t _attachment = 0)
	{
		if (0 != (BGFX_CLEAR_COLOR_USE_PALETTE & _clear.m_flags) )
		{
			const uint8_t index = bx::min<uint8_t>(BGFX_CONFIG_MAX_COLOR_PALETTE-1, _clear.m_index[_attachment]);
			const float* rgba = _palette[index];
			return { rgba[0], rgba[1], rgba[2], rgba[3] };
		}

		return
		{
			_clear.m_index[0]/255.0f,
			_clear.m_index[1]/255.0f,
			_clear.m_index[2]/255.0f,
			_clear.m_index[3]/255.0f,
		};
	}

	static const char* s_attribName[] =
	{
		"a_position",
		"a_normal",
		"a_tangent",
		"a_bitangent",
		"a_color0",
		"a_color1",
		"a_color2",
		"a_color3",
		"a_indices",
		"a_weight",
		"a_texcoord0",
		"a_texcoord1",
		"a_texcoord2",
		"a_texcoord3",
		"a_texcoord4",
		"a_texcoord5",
		"a_texcoord6",
		"a_texcoord7",
		"a_texcoord8",
		"a_texcoord9",
		"a_texcoord10",
		"a_texcoord11",
		"a_texcoord12",
		"a_texcoord13",
		"a_texcoord14",
		"a_texcoord15",
	};
	static_assert(Attrib::Count == BX_COUNTOF(s_attribName) );

	static const GPU::CompareFunction s_cmpFunc[] =
	{
		GPU::CompareFunction::Always,
		GPU::CompareFunction::Less,
		GPU::CompareFunction::LessEqual,
		GPU::CompareFunction::Equal,
		GPU::CompareFunction::GreaterEqual,
		GPU::CompareFunction::Greater,
		GPU::CompareFunction::NotEqual,
		GPU::CompareFunction::Never,
		GPU::CompareFunction::Always,
	};

	static const GPU::StencilOp s_stencilOp[] =
	{
		GPU::StencilOp::Zero,
		GPU::StencilOp::Keep,
		GPU::StencilOp::Replace,
		GPU::StencilOp::IncrementWrap,
		GPU::StencilOp::IncrementClamp,
		GPU::StencilOp::DecrementWrap,
		GPU::StencilOp::DecrementClamp,
		GPU::StencilOp::Invert,
	};

	// [factor][0 = colour, 1 = alpha], as renderer_mtl.cpp's s_blendFactor. eacp has
	// no constant blend colour (G26), so FACTOR and INV_FACTOR stand in as One and
	// Zero.
	static const GPU::BlendFactor s_blendFactor[][2] =
	{
		{ GPU::BlendFactor::Zero,                     GPU::BlendFactor::Zero                     }, // ignored
		{ GPU::BlendFactor::Zero,                     GPU::BlendFactor::Zero                     }, // ZERO
		{ GPU::BlendFactor::One,                      GPU::BlendFactor::One                      }, // ONE
		{ GPU::BlendFactor::SourceColor,              GPU::BlendFactor::SourceAlpha              }, // SRC_COLOR
		{ GPU::BlendFactor::OneMinusSourceColor,      GPU::BlendFactor::OneMinusSourceAlpha      }, // INV_SRC_COLOR
		{ GPU::BlendFactor::SourceAlpha,              GPU::BlendFactor::SourceAlpha              }, // SRC_ALPHA
		{ GPU::BlendFactor::OneMinusSourceAlpha,      GPU::BlendFactor::OneMinusSourceAlpha      }, // INV_SRC_ALPHA
		{ GPU::BlendFactor::DestinationAlpha,         GPU::BlendFactor::DestinationAlpha         }, // DST_ALPHA
		{ GPU::BlendFactor::OneMinusDestinationAlpha, GPU::BlendFactor::OneMinusDestinationAlpha }, // INV_DST_ALPHA
		{ GPU::BlendFactor::DestinationColor,         GPU::BlendFactor::DestinationAlpha         }, // DST_COLOR
		{ GPU::BlendFactor::OneMinusDestinationColor, GPU::BlendFactor::OneMinusDestinationAlpha }, // INV_DST_COLOR
		{ GPU::BlendFactor::SourceAlphaSaturated,     GPU::BlendFactor::One                      }, // SRC_ALPHA_SAT
		{ GPU::BlendFactor::One,                      GPU::BlendFactor::One                      }, // FACTOR (G26)
		{ GPU::BlendFactor::Zero,                     GPU::BlendFactor::Zero                     }, // INV_FACTOR (G26)
	};

	static const GPU::BlendOperation s_blendEquation[] =
	{
		GPU::BlendOperation::Add,
		GPU::BlendOperation::Subtract,
		GPU::BlendOperation::ReverseSubtract,
		GPU::BlendOperation::Min,
		GPU::BlendOperation::Max,
	};

	struct PrimInfo
	{
		GPU::PrimitiveTopology m_type;
		uint32_t m_min;
		uint32_t m_div;
		uint32_t m_sub;
	};

	static const PrimInfo s_primInfo[] =
	{
		{ GPU::PrimitiveTopology::Triangles,     3, 3, 0 },
		{ GPU::PrimitiveTopology::TriangleStrip, 3, 1, 2 },
		{ GPU::PrimitiveTopology::Lines,         2, 2, 0 },
		{ GPU::PrimitiveTopology::LineStrip,     2, 1, 1 },
		{ GPU::PrimitiveTopology::Points,        1, 1, 0 },
	};
	static_assert(Topology::Count == BX_COUNTOF(s_primInfo) );

	// bgfx's attribute type, count and normalisation as the nearest eacp vertex
	// format. What eacp has: Float x1..4, UByte4Norm, Half2/4, Short2/4Norm, and
	// no integer format at all. The rest (non-normalised integers, Uint10) is G7.
	// A non-normalised one is given the normalised format of its size: right for
	// a shader input declared float (it reads 0..1 rather than 0..255), refused
	// by Metal when the input is declared integer (uvec4 a_indices), in which
	// case the pipeline is invalid and its draws are skipped.
	static GPU::VertexFormat toVertexFormat(uint8_t _num, AttribType::Enum _type, bool _normalized, bool _asInt)
	{
		// asInt is ignored, as renderer_mtl.cpp's s_attribType ignores it: bgfx's
		// meshes pack normals as Uint8 normalized asInt, and the shader unpacks.
		BX_UNUSED(_asInt);

		switch (_type)
		{
		case AttribType::Uint8:
			if (!_normalized)
			{
				traceOnce(Unsupported::VertexFormat, "G7: a non-normalised Uint8 attribute is given UByte4Norm; a float input reads it scaled to 0..1, an integer input (uvec4) fails the pipeline and its draws are skipped.");
			}
			return GPU::VertexFormat::UByte4Norm;

		case AttribType::Uint10:
			traceOnce(Unsupported::VertexFormat, "G7: a Uint10 attribute is given UByte4Norm, which reads the wrong bits.");
			return GPU::VertexFormat::UByte4Norm;

		case AttribType::Int16:
			if (!_normalized)
			{
				traceOnce(Unsupported::VertexFormat, "G7: a non-normalised Int16 attribute is given ShortNorm; a float input reads it scaled to -1..1, an integer input fails the pipeline and its draws are skipped.");
			}
			return _num <= 2 ? GPU::VertexFormat::Short2Norm : GPU::VertexFormat::Short4Norm;

		case AttribType::Half:
			return _num <= 2 ? GPU::VertexFormat::Half2 : GPU::VertexFormat::Half4;

		case AttribType::Float:
		default:
			break;
		}

		switch (_num)
		{
		case 1:  return GPU::VertexFormat::Float;
		case 2:  return GPU::VertexFormat::Float2;
		case 3:  return GPU::VertexFormat::Float3;
		default: return GPU::VertexFormat::Float4;
		}
	}

	/// A bgfx vertex or index buffer. Static ones are one device buffer. Dynamic
	/// ones (bgfx's dynamic and transient buffers, written by update* between
	/// frames and, for the debug text, between draws of one frame) keep a CPU copy
	/// and are drawn from a ring of one streaming buffer per frame in flight, so a
	/// write never lands on bytes a frame still on the GPU reads; a second write
	/// in a frame that already drew from its slot goes to a StreamingBuffers
	/// slice instead.
	///
	/// A buffer a kernel reads or writes (BGFX_BUFFER_COMPUTE_*, and indirect
	/// buffers, which kernels fill) is one device buffer of BufferUsage::Storage
	/// whatever bgfx calls it, static or dynamic: the GPU's writes have to land
	/// in the bytes the next draw or dispatch reads, which a ring of CPU-filled
	/// copies cannot give. On Metal any buffer binds as vertex, index or
	/// storage, so the one buffer serves all three. bgfx refuses CPU updates of
	/// a COMPUTE_WRITE buffer; a COMPUTE_READ one is updated in place through
	/// Buffer::update, which orders the write after the frames already
	/// submitted (it waits for them on Metal).
	struct BufferEacp
	{
		static constexpr int kNumSlots = GPU::StreamingBuffers::framesInFlight;

		BufferEacp(uint32_t _size, const void* _data, uint16_t _flags, bool _vertex, VertexLayoutHandle _layoutHandle)
			: m_size(_size)
			, m_flags(_flags)
			, m_vertex(_vertex)
			, m_storage(0 != (_flags & (BGFX_BUFFER_COMPUTE_READ_WRITE|BGFX_BUFFER_DRAW_INDIRECT) ) )
			, m_layoutHandle(_layoutHandle)
		{
			const uint32_t bytes = bx::max<uint32_t>(_size, 4);

			if (NULL != _data)
			{
				m_static.emplace(GPU::Device::shared().makeBuffer(_data, bytes, usage() ) );
			}
			else if (m_storage)
			{
				// Zeroed rather than uninitialised: a kernel that accumulates
				// into it, or an indirect buffer read before it is written, sees
				// what a fresh buffer on every other bgfx backend holds.
				const std::vector<uint8_t> zero(bytes, 0);
				m_static.emplace(GPU::Device::shared().makeBuffer(zero.data(), bytes, usage() ) );
			}
			else
			{
				m_shadow.resize(bytes, 0);
			}
		}

		GPU::BufferUsage usage() const
		{
			return m_storage ? GPU::BufferUsage::Storage
				: m_vertex   ? GPU::BufferUsage::Vertex
				:              GPU::BufferUsage::Index
				;
		}

		void update(uint32_t _offset, uint32_t _size, const void* _data)
		{
			if (m_static
			&&  m_storage)
			{
				if (_offset < m_size)
				{
					m_static->update(_data, bx::min(_size, m_size - _offset), _offset);
				}

				return;
			}

			if (m_static)
			{
				BX_TRACE("eacp: update of a static buffer ignored.");
				return;
			}

			if (_offset >= m_size)
			{
				return;
			}

			_size = bx::min(_size, m_size - _offset);
			bx::memCopy(&m_shadow[_offset], _data, _size);
			m_highWater = bx::max(m_highWater, _offset + _size);
			++m_version;
		}

		/// The range a draw in the current frame reads this buffer from.
		GPU::BufferRange range(GPU::StreamingBuffers& _stream)
		{
			if (m_static)
			{
				return GPU::BufferRange::of(*m_static);
			}

			const uint64_t frame = GPU::Device::shared().frameIndex();
			Slot& slot = m_slots[frame % kNumSlots];

			if (slot.m_buffer
			&&  slot.m_version == m_version)
			{
				slot.m_frame = frame;
				return GPU::BufferRange::of(*slot.m_buffer);
			}

			const uint32_t bytes = bx::max<uint32_t>(m_highWater, 4);

			if (slot.m_buffer
			&&  slot.m_frame == frame)
			{
				// Drawn from already in this frame, with the contents before this
				// write: those draws keep the slot, this one gets a slice.
				return _stream.write(m_shadow.data(), bytes);
			}

			if (!slot.m_buffer)
			{
				slot.m_buffer.emplace(GPU::Device::shared().makeBuffer(
					  m_shadow.data()
					, int64_t(m_shadow.size() )
					, usage()
					, GPU::BufferStorage::Streaming
					) );
			}
			else
			{
				// The slot was last read by the frame kNumSlots ago, which is no
				// longer on the GPU: StreamingBuffers' own guarantee.
				slot.m_buffer->updateUnordered(m_shadow.data(), bytes, 0);
			}

			slot.m_version = m_version;
			slot.m_frame   = frame;
			return GPU::BufferRange::of(*slot.m_buffer);
		}

		struct Slot
		{
			std::optional<GPU::Buffer> m_buffer;
			uint64_t m_version = UINT64_MAX;
			uint64_t m_frame   = UINT64_MAX;
		};

		uint32_t m_size;
		uint16_t m_flags;
		bool m_vertex;
		bool m_storage; // one device buffer kernels read and write
		VertexLayoutHandle m_layoutHandle;

		std::optional<GPU::Buffer> m_static;
		std::vector<uint8_t> m_shadow;
		uint32_t m_highWater = 0;
		uint64_t m_version   = 0;
		Slot m_slots[kNumSlots];
	};

	struct ShaderEacp
	{
		ShaderEacp()
			: m_constantBuffer(NULL)
			, m_magic(0)
			, m_hash(0)
			, m_size(0)
			, m_numPredefined(0)
			, m_numSamplers(0)
			, m_numStorage(0)
		{
		}

		~ShaderEacp()
		{
			if (NULL != m_constantBuffer)
			{
				UniformBuffer::destroy(m_constantBuffer);
				m_constantBuffer = NULL;
			}
		}

		bool create(const Memory* _mem, const UniformRegistry& _uniformReg)
		{
			bx::MemoryReader reader(_mem->data, _mem->size);

			bx::Error err;

			bx::read(&reader, m_magic, &err);

			const bool fragment = isShaderType(m_magic, 'F');
			const uint8_t fragmentBit = fragment ? kUniformFragmentBit : 0;

			uint32_t hashIn;
			bx::read(&reader, hashIn, &err);

			uint32_t hashOut;
			bx::read(&reader, hashOut, &err);

			{
				uint32_t rawSrvMask, rawUavMask;
				readRawBindings(&reader, rawSrvMask, rawUavMask, &err);
				BX_UNUSED(rawSrvMask, rawUavMask);
			}

			uint16_t count = 0;
			bx::read(&reader, count, &err);

			BX_TRACE("eacp: %s shader, %d uniforms", getShaderTypeName(m_magic), count);

			uint32_t blockSize = 0;

			for (uint32_t ii = 0; ii < count && err.isOk(); ++ii)
			{
				uint8_t nameSize = 0;
				bx::read(&reader, nameSize, &err);

				char name[256];
				bx::read(&reader, &name, nameSize, &err);
				name[nameSize] = '\0';

				uint8_t type = 0;
				bx::read(&reader, type, &err);

				uint8_t num;
				bx::read(&reader, num, &err);

				uint16_t regIndex;
				bx::read(&reader, regIndex, &err);

				uint16_t regCount;
				bx::read(&reader, regCount, &err);

				uint8_t  texComponent;
				uint8_t  texDimension;
				uint16_t texFormat;
				bx::read(&reader, texComponent, &err);
				bx::read(&reader, texDimension, &err);
				bx::read(&reader, texFormat, &err);
				BX_UNUSED(texComponent, texFormat);

				const uint8_t baseType = type & ~kUniformMask;
				const char* kind = "user";

				if (UniformType::Sampler == baseType
				||  0 != (type & kUniformSamplerBit) )
				{
					// eacp-metal writes the texture index, which is the bind
					// stage: SAMPLER2D(s_tex, n) declares [[texture(n)]] and
					// [[sampler(n)]], and setTexture(n, ...) fills m_bind[n].
					kind = "sampler";

					if (m_numSamplers < BX_COUNTOF(m_samplers) )
					{
						m_samplers[m_numSamplers].m_stage     = uint8_t(regIndex);
						m_samplers[m_numSamplers].m_dimension = texDimension;
						m_numSamplers++;
					}
				}
				else if (UniformType::End == baseType
				     &&  (bx::hasSuffix(name, "Texture") || bx::hasSuffix(name, "Sampler") || bx::hasSuffix(name, "SamplerComparison") ) )
				{
					// glslang's live uniforms list a sampler's two halves
					// (s_texTexture, s_texSampler, or s_texSamplerComparison
					// for a shadow sampler) at offset 0xffff; the Sampler
					// entry above is the one that matters.
					kind = "sampler half";
				}
				else if (UniformType::End == baseType)
				{
					// A storage image or buffer: step 6.
					kind = "storage";
					++m_numStorage;
				}
				else
				{
					// regIndex is the byte offset into the stage's block, as on
					// Vulkan: shaderc_metal.cpp writes the member's offset.
					// An array's registers by its element count as well: the
					// clear quad's bgfx_clear_color[8] is listed with one
					// register and is 128 bytes in the MSL block.
					const uint32_t typeRegs = UniformType::Mat4 == baseType ? 4
						: UniformType::Mat3 == baseType ? 3
						: 1
						;
					blockSize = bx::max<uint32_t>(blockSize, regIndex + bx::max<uint32_t>(regCount, num*typeRegs)*16);

					const PredefinedUniform::Enum predefined = nameToPredefinedUniformEnum(name);

					if (PredefinedUniform::Count != predefined)
					{
						kind = "predefined";
						m_predefined[m_numPredefined].m_loc   = regIndex;
						m_predefined[m_numPredefined].m_count = regCount;
						m_predefined[m_numPredefined].m_type  = uint8_t(predefined|fragmentBit);
						m_numPredefined++;
					}
					else if (const UniformRegInfo* info = _uniformReg.find(name) )
					{
						if (NULL == m_constantBuffer)
						{
							m_constantBuffer = UniformBuffer::create(1024);
						}

						m_constantBuffer->writeUniformHandle(baseType|fragmentBit, regIndex, info->m_handle, regCount);
					}
					else
					{
						// The clear quad's own uniforms, which the backend writes
						// into the block directly.
						kind = "unregistered";
					}
				}

				BX_TRACE("eacp: \t%s: %s (%s), offset %d, regs %d"
					, kind
					, name
					, getUniformTypeName(UniformType::Enum(baseType) )
					, regIndex
					, regCount
					);
				BX_UNUSED(kind);
			}

			if (NULL != m_constantBuffer)
			{
				m_constantBuffer->finish();
			}

			if (isShaderType(m_magic, 'C') )
			{
				// The kernel's threadgroup, which bgfx's dispatch counts
				// groups of.
				bx::read(&reader, m_numThreads, sizeof(m_numThreads), &err);
			}

			uint32_t shaderSize = 0;
			bx::read(&reader, shaderSize, &err);

			if (!err.isOk()
			||  shaderSize > reader.remaining() )
			{
				BX_TRACE("eacp: corrupted shader binary.");
				return false;
			}

			const char* code = (const char*)reader.getDataPtr();
			m_code.assign(code, shaderSize);
			bx::skip(&reader, shaderSize+1);

			uint8_t numAttrs = 0;
			bx::read(&reader, numAttrs, &err);
			bx::skip(&reader, numAttrs*sizeof(uint16_t) );

			uint16_t size = 0;
			bx::read(&reader, size, &err);
			m_size = bx::max<uint32_t>(bx::max<uint32_t>(size, blockSize), globalBlockSize(m_code) );

			if (!isShaderType(m_magic, 'C') )
			{
				m_numStorage = uint8_t(bx::max<uint32_t>(m_numStorage, renderStorageCount(m_code) ) );
			}

			bx::HashMurmur3 murmur;
			murmur.begin();
			murmur.add(hashIn);
			murmur.add(hashOut);
			murmur.add(code, shaderSize);
			m_hash = murmur.end();

			return true;
		}

		/// How many render-stage storage buffers the MSL declares: eacp-metal
		/// puts BUFFER_RO/RW at stage n on [[buffer(bufferBase + n)]], and the
		/// uniform table does not always list them (41-tess's
		/// vs_terrain_render reads three and lists none).
		static uint32_t renderStorageCount(const std::string& _msl)
		{
			static const char s_tag[] = "[[buffer(";
			uint32_t count = 0;

			for (size_t pos = _msl.find(s_tag); std::string::npos != pos; pos = _msl.find(s_tag, pos) )
			{
				pos += sizeof(s_tag) - 1;

				if (atoi(_msl.c_str() + pos) >= GPU::RenderPass::bufferBase)
				{
					++count;
				}
			}

			return count;
		}

		/// The size of the MSL `struct _Global`, which the bound bytes must
		/// cover. The uniform table can say less: it lists the elements a
		/// shader uses, so fs_clear0's `float4 bgfx_clear_color[8]` is one
		/// register there and 128 bytes in the block. bgfx's uniforms are
		/// vec4, mat3 and mat4, each 16-byte aligned.
		static uint32_t globalBlockSize(const std::string& _msl)
		{
			const size_t begin = _msl.find("struct _Global");
			if (std::string::npos == begin)
			{
				return 0;
			}

			const size_t open  = _msl.find('{', begin);
			const size_t close = _msl.find('}', begin);
			if (std::string::npos == open
			||  std::string::npos == close)
			{
				return 0;
			}

			uint32_t bytes = 0;
			size_t pos = open + 1;

			while (pos < close)
			{
				size_t end = _msl.find(';', pos);
				if (std::string::npos == end
				||  end > close)
				{
					break;
				}

				const std::string member = _msl.substr(pos, end-pos);
				pos = end + 1;

				const uint32_t element = std::string::npos != member.find("float4x4") ? 64
					: std::string::npos != member.find("float3x3") ? 48
					: std::string::npos != member.find("float2x2") ? 32
					: 16
					;

				uint32_t count = 1;
				const size_t bracket = member.find('[');
				if (std::string::npos != bracket)
				{
					count = bx::max<uint32_t>(1, uint32_t(atoi(member.c_str() + bracket + 1) ) );
				}

				bytes += element*count;
			}

			return bytes;
		}

		std::string m_code;
		UniformBuffer* m_constantBuffer;
		PredefinedUniform m_predefined[PredefinedUniform::Count];
		uint32_t m_magic;
		uint32_t m_hash;
		uint32_t m_size;
		uint8_t m_numPredefined;

		/// A texture the stage samples: the bind stage (and texture/sampler
		/// index) and shaderc's TextureDimension id.
		struct Sampler
		{
			uint8_t m_stage;
			uint8_t m_dimension;
		};

		Sampler m_samplers[BGFX_CONFIG_MAX_TEXTURE_SAMPLERS];
		uint8_t m_numSamplers;
		uint8_t m_numStorage;
		uint16_t m_numThreads[3] = { 1, 1, 1 };
	};

	/// Where the vertex stage's [[attribute(n)]] comes from.
	struct AttributeSource
	{
		enum Enum : uint8_t
		{
			Vertex,
			Instance,
			Unknown,
		};

		Enum    m_kind  = Unknown;
		uint8_t m_index = 0; // Attrib::Enum or the i_dataN number
	};

	static constexpr uint32_t kMaxAttributes = 31;

	struct ProgramEacp
	{
		ProgramEacp()
			: m_vsh(NULL)
			, m_fsh(NULL)
			, m_numPredefined(0)
			, m_numAttributes(0)
		{
		}

		static bool isPreamble(const std::string& _line)
		{
			return 0 == _line.rfind("#include", 0)
				|| 0 == _line.rfind("using namespace", 0)
				;
		}

		/// G23: one ShaderLibrary per pipeline, and bgfx's two stages are two MSL
		/// translation units whose names clash (_Global, xlatMtlMain_out). Each
		/// goes into a namespace of its own, with the includes hoisted out.
		static std::string programSource(const std::string& _vertex, const std::string& _fragment)
		{
			std::string preamble;
			std::string body[2];
			const std::string* stage[2] = { &_vertex, &_fragment };

			for (uint32_t ii = 0; ii < 2; ++ii)
			{
				size_t pos = 0;

				while (pos < stage[ii]->size() )
				{
					size_t end = stage[ii]->find('\n', pos);
					end = std::string::npos == end ? stage[ii]->size() : end;

					const std::string line = stage[ii]->substr(pos, end-pos) + "\n";

					if (isPreamble(line) )
					{
						if (std::string::npos == preamble.find(line) )
						{
							preamble += line;
						}
					}
					else
					{
						body[ii] += line;
					}

					pos = end + 1;
				}
			}

			return preamble
				+ "namespace bgfx_vs\n{\n" + body[0] + "} // namespace bgfx_vs\n"
				+ "namespace bgfx_fs\n{\n" + body[1] + "} // namespace bgfx_fs\n"
				;
		}

		/// The vertex inputs as the MSL declares them: `<type> <name>
		/// [[attribute(<n>)]];` in xlatMtlMain_in. The binary's attribute list says
		/// the same for vertex attributes, but names every i_dataN 0xffff.
		void parseAttributes(const std::string& _msl)
		{
			m_numAttributes = 0;

			const std::string tag = "[[attribute(";
			size_t pos = 0;

			while (std::string::npos != (pos = _msl.find(tag, pos) ) )
			{
				const uint32_t location = uint32_t(atoi(_msl.c_str() + pos + tag.size() ) );

				size_t nameEnd = pos;
				while (nameEnd > 0 && ' ' == _msl[nameEnd-1])
				{
					--nameEnd;
				}

				size_t nameBegin = nameEnd;
				while (nameBegin > 0
				&&    (bx::isAlphaNum(_msl[nameBegin-1]) || '_' == _msl[nameBegin-1]) )
				{
					--nameBegin;
				}

				pos += tag.size();

				if (location >= kMaxAttributes)
				{
					continue;
				}

				const std::string name = _msl.substr(nameBegin, nameEnd-nameBegin);
				AttributeSource& source = m_attributes[location];
				source.m_kind = AttributeSource::Unknown;

				for (uint8_t ii = 0; ii < Attrib::Count; ++ii)
				{
					if (name == s_attribName[ii])
					{
						source.m_kind  = AttributeSource::Vertex;
						source.m_index = ii;
					}
				}

				if (0 == name.rfind("i_data", 0) )
				{
					source.m_kind  = AttributeSource::Instance;
					source.m_index = uint8_t(atoi(name.c_str() + 6) );
				}

				m_numAttributes = bx::max<uint32_t>(m_numAttributes, location+1);
			}
		}

		bool create(const ShaderEacp* _vsh, const ShaderEacp* _fsh)
		{
			m_vsh = _vsh;
			m_fsh = _fsh;

			bx::memCopy(&m_predefined[0], _vsh->m_predefined, _vsh->m_numPredefined*sizeof(PredefinedUniform) );
			m_numPredefined = _vsh->m_numPredefined;

			if (NULL != _fsh)
			{
				bx::memCopy(&m_predefined[m_numPredefined], _fsh->m_predefined, _fsh->m_numPredefined*sizeof(PredefinedUniform) );
				m_numPredefined += _fsh->m_numPredefined;
			}

			if (isShaderType(_vsh->m_magic, 'C') )
			{
				return createCompute(*_vsh);
			}

			parseAttributes(_vsh->m_code);

			if (NULL == _fsh)
			{
				return createVertexOnly(*_vsh);
			}

			GPU::ShaderSource source = GPU::ShaderSource::msl(programSource(_vsh->m_code, _fsh->m_code) );
			source.withVertex("bgfx_vs::xlatMtlMain");
			source.withFragment("bgfx_fs::xlatMtlMain");

			m_library.emplace(GPU::Device::shared().makeShaderLibrary(source) );

			if (!m_library->isValid() )
			{
				BX_TRACE("eacp: the program's MSL does not compile.");
				m_library.reset();
				return false;
			}

			return true;
		}

		/// createProgram(vsh, invalid): a depth or stencil only program. One
		/// stage, so no namespaces, and a library without a fragment entry
		/// (G56), whose pipelines write no colour (writtenColorTarget) while
		/// still naming the pass's colour formats.
		bool createVertexOnly(const ShaderEacp& _vsh)
		{
			GPU::ShaderSource source = GPU::ShaderSource::msl(_vsh.m_code);
			source.withVertex("xlatMtlMain");
			source.withFragment({});

			m_library.emplace(GPU::Device::shared().makeShaderLibrary(source) );

			if (!m_library->isValid() )
			{
				BX_TRACE("eacp: the vertex-only program's MSL does not compile.");
				m_library.reset();
				return false;
			}

			return true;
		}

		/// One stage, so no namespaces: the kernel is the MSL as shaderc wrote
		/// it, entry xlatMtlMain, with the threadgroup from the binary, which
		/// eacp's ComputePass dispatches every grid in.
		bool createCompute(const ShaderEacp& _csh)
		{
			GPU::ShaderSource source = GPU::ShaderSource::msl(_csh.m_code);
			source.withCompute("xlatMtlMain");
			source.withThreadGroup({ int(_csh.m_numThreads[0]), int(_csh.m_numThreads[1]), int(_csh.m_numThreads[2]) });

			m_library.emplace(GPU::Device::shared().makeShaderLibrary(source) );

			if (!m_library->isValid() )
			{
				BX_TRACE("eacp: the kernel's MSL does not compile.");
				m_library.reset();
				return false;
			}

			m_compute.emplace(GPU::Device::shared().makeComputePipeline(*m_library) );

			if (!m_compute->isValid() )
			{
				BX_TRACE("eacp: the kernel's pipeline does not build.");
				m_compute.reset();
				m_library.reset();
				return false;
			}

			BX_TRACE("eacp: kernel, threadgroup %dx%dx%d.", _csh.m_numThreads[0], _csh.m_numThreads[1], _csh.m_numThreads[2]);
			return true;
		}

		bool isCompute() const
		{
			return m_compute.has_value();
		}

		bool usesStorage() const
		{
			return 0 != m_vsh->m_numStorage
				|| (NULL != m_fsh && 0 != m_fsh->m_numStorage)
				;
		}

		std::optional<GPU::ShaderLibrary> m_library;
		std::optional<GPU::ComputePipeline> m_compute;
		const ShaderEacp* m_vsh;
		const ShaderEacp* m_fsh;
		PredefinedUniform m_predefined[PredefinedUniform::Count*2];
		uint8_t m_numPredefined;
		uint32_t m_numAttributes;
		AttributeSource m_attributes[kMaxAttributes];
	};

	/// bgfx's texture format as the eacp one, for what eacp has (G4 is the
	/// rest). A format without one is decoded to BGRA8 when bimg can, which is
	/// what g_caps' _EMULATED bits (set by bgfx itself) promise. Every format
	/// here is stored as itself; only depth is not, every depth format being
	/// a 32-bit float plane.
	static bool toEacpFormat(TextureFormat::Enum _format, GPU::TextureFormat& _out)
	{
		switch (_format)
		{
		case TextureFormat::R16F:    _out = GPU::TextureFormat::R16Float;    return true;
		case TextureFormat::RG16F:   _out = GPU::TextureFormat::RG16Float;   return true;
		case TextureFormat::R16:     _out = GPU::TextureFormat::R16Unorm;    return true;
		case TextureFormat::RG32F:   _out = GPU::TextureFormat::RG32Float;   return true;
		case TextureFormat::R32U:    _out = GPU::TextureFormat::R32Uint;     return true;
		case TextureFormat::BC1:     _out = GPU::TextureFormat::BC1RGBA;     return true;
		case TextureFormat::BC2:     _out = GPU::TextureFormat::BC2RGBA;     return true;
		case TextureFormat::BC3:     _out = GPU::TextureFormat::BC3RGBA;     return true;
		case TextureFormat::BC7:     _out = GPU::TextureFormat::BC7RGBA;     return true;
		case TextureFormat::R8:      _out = GPU::TextureFormat::R8Unorm;     return true;
		case TextureFormat::RG8:     _out = GPU::TextureFormat::RG8Unorm;    return true;
		case TextureFormat::BGRA8:   _out = GPU::TextureFormat::BGRA8Unorm;  return true;
		case TextureFormat::RGBA8:   _out = GPU::TextureFormat::RGBA8Unorm;  return true;
		case TextureFormat::RGBA16F: _out = GPU::TextureFormat::RGBA16Float; return true;
		case TextureFormat::RGBA32F: _out = GPU::TextureFormat::RGBA32Float; return true;
		case TextureFormat::R32F:    _out = GPU::TextureFormat::R32Float;    return true;
		default:
			break;
		}

		return false;
	}

	/// Whether eacp renders into the texture format: every uncompressed one
	/// has its PixelFormat (pixelFormatFor), R32Uint as an integer attachment
	/// that takes no blending.
	static bool isRenderable(GPU::TextureFormat _format)
	{
		return !GPU::isCompressedFormat(_format);
	}

	/// The sample count of a BGFX_TEXTURE_RT_MSAA_* render target: the count
	/// asked for, or the highest below it the device multisamples at, as
	/// renderer_mtl.cpp's choice from its sample count table. eacp refuses a
	/// count the device cannot take rather than clamping it.
	static int sampleCountFor(uint64_t _flags)
	{
		const uint32_t msaa = uint32_t( (_flags&BGFX_TEXTURE_RT_MSAA_MASK)>>BGFX_TEXTURE_RT_MSAA_SHIFT);

		int count = 1 < msaa ? int(1 << bx::min<uint32_t>(msaa-1, 4) ) : 1;

		while (1 < count
		&&     !GPU::Device::shared().supportsSampleCount(count) )
		{
			count >>= 1;
		}

		return count;
	}

	static uint32_t mipExtent(uint32_t _extent, uint8_t _mip)
	{
		return bx::max<uint32_t>(1, _extent >> _mip);
	}

	/// The eacp kind a bgfx texture is: bgfx's layers and cube flag, or a
	/// volume.
	static GPU::TextureType textureKind(bool _cube, uint16_t _numLayers, bool _volume)
	{
		if (_volume)
		{
			return GPU::TextureType::ThreeD;
		}

		if (_cube)
		{
			return 1 < _numLayers ? GPU::TextureType::CubeArray : GPU::TextureType::Cube;
		}

		return 1 < _numLayers ? GPU::TextureType::TwoDArray : GPU::TextureType::TwoD;
	}

	/// The eacp kind a shader's sampler declares (shaderc's TextureDimension
	/// id; 1D and unknown are sampled as 2D).
	static GPU::TextureType samplerKind(uint8_t _dimension)
	{
		switch (_dimension)
		{
		case 0x03: return GPU::TextureType::TwoDArray; // TextureDimension::Dimension2DArray
		case 0x04: return GPU::TextureType::Cube;      // TextureDimension::DimensionCube
		case 0x05: return GPU::TextureType::CubeArray; // TextureDimension::DimensionCubeArray
		case 0x06: return GPU::TextureType::ThreeD;    // TextureDimension::Dimension3D
		default:   return GPU::TextureType::TwoD;
		}
	}

	/// A bgfx texture: one eacp texture of any kind (2D, cube, 2D array, cube
	/// array, volume), created from every level of every slice bimg's
	/// container holds, and updated and read one subresource at a time
	/// (Texture::updateSubresource, readSubresource). A slice is eacp's: a
	/// layer, a face, layer*6 + face on a cube array, which is also bimg's
	/// side index into a container.
	struct TextureEacp
	{
		TextureEacp()
			: m_flags(0)
			, m_width(0)
			, m_height(0)
			, m_depth(1)
			, m_numLayers(1)
			, m_numMips(1)
			, m_numSides(1)
			, m_requestedFormat(TextureFormat::Unknown)
			, m_textureFormat(TextureFormat::Unknown)
			, m_format(GPU::TextureFormat::RGBA8Unorm)
			, m_kind(GPU::TextureType::TwoD)
			, m_cube(false)
			, m_depthFormat(false)
			, m_stencilFormat(false)
			, m_depthKept(false)
			, m_depthOf(kInvalidHandle)
		{
		}

		/// The eacp texture a render pass draws into. A depth texture has no eacp
		/// object of its own; when a framebuffer pairs it with no colour, it
		/// gets a colour target of its own size to carry its depth plane.
		bool isTarget() const
		{
			return m_texture
				&& m_texture->isRenderTarget()
				;
		}

		bool isVolume() const
		{
			return GPU::TextureType::ThreeD == m_kind;
		}

		/// Recreates the render target with (or without) a depth plane: eacp's
		/// depth is a property of the colour target, decided when it is made
		/// (TextureDescriptor::depth), so a framebuffer that pairs a colour
		/// texture with a depth texture remakes the colour one. Its contents
		/// are lost, which is nothing for a texture no pass has drawn into yet.
		bool makeTarget(bool _depth, bool _stencil, bool _sampleable)
		{
			GPU::TextureDescriptor desc = m_desc;
			desc.depth           = _depth || _stencil || _sampleable;
			desc.stencil         = _stencil;
			desc.sampleableDepth = _sampleable;

			if (m_texture
			&&  m_texture->hasDepth()          == desc.depth
			&&  m_texture->hasStencil()        == desc.stencil
			&&  m_texture->hasSampleableDepth() == desc.sampleableDepth)
			{
				return true;
			}

			m_texture.emplace(GPU::Device::shared().makeTexture(desc, NULL) );
			m_depthKept = false;

			if (!m_texture->isValid() )
			{
				BX_TRACE("eacp: makeTexture refused the %dx%d render target.", desc.width, desc.height);
				m_texture.reset();
				return false;
			}

			m_desc = desc;
			return true;
		}

		uint32_t levelBytes(uint8_t _mip) const
		{
			return uint32_t(GPU::levelBytes(m_format, int(mipExtent(m_width, _mip) ), int(mipExtent(m_height, _mip) ) ) );
		}

		uint32_t slices() const
		{
			return isVolume() ? 1 : uint32_t(m_numLayers)*m_numSides;
		}

		void create(const Memory* _mem, uint64_t _flags, uint8_t _skip)
		{
			destroy();

			bimg::ImageContainer imageContainer;

			if (!bimg::imageParse(imageContainer, _mem->data, _mem->size) )
			{
				BX_TRACE("eacp: texture data does not parse.");
				return;
			}

			const uint8_t startLod = bx::min<uint8_t>(_skip, imageContainer.m_numMips-1);
			const bool    volume   = bimg::isVolume(imageContainer);

			bimg::TextureInfo ti;
			bimg::imageGetSize(
				  &ti
				, uint16_t(imageContainer.m_width >>startLod)
				, uint16_t(imageContainer.m_height>>startLod)
				, uint16_t(imageContainer.m_depth >>startLod)
				, imageContainer.m_cubeMap
				, 1 < imageContainer.m_numMips
				, imageContainer.m_numLayers
				, imageContainer.m_format
				);
			ti.numMips = bx::min<uint8_t>(imageContainer.m_numMips-startLod, ti.numMips);

			m_flags           = _flags;
			m_width           = ti.width;
			m_height          = ti.height;
			m_depth           = volume ? bx::max<uint32_t>(1, ti.depth) : 1;
			m_numLayers       = bx::max<uint16_t>(1, ti.numLayers);
			m_numMips         = ti.numMips;
			m_cube            = imageContainer.m_cubeMap;
			m_numSides        = m_cube ? 6 : 1;
			m_kind            = textureKind(m_cube, m_numLayers, volume);
			m_requestedFormat = TextureFormat::Enum(imageContainer.m_format);
			m_textureFormat   = getViableTextureFormat(imageContainer);

			const bool renderTarget = 0 != (_flags & BGFX_TEXTURE_RT_MASK);
			const bool computeWrite = 0 != (_flags & BGFX_TEXTURE_COMPUTE_WRITE);

			BX_TRACE("eacp: texture %s (requested %s), %dx%dx%d%s, layers %d, mips %d, RT[%c] CW[%c] sRGB[%c]"
				, getName(m_textureFormat)
				, getName(m_requestedFormat)
				, ti.width
				, ti.height
				, m_depth
				, m_cube ? "x6" : ""
				, ti.numLayers
				, ti.numMips
				, renderTarget ? 'x' : ' '
				, computeWrite ? 'x' : ' '
				, 0 != (_flags & BGFX_TEXTURE_SRGB) ? 'x' : ' '
				);

			if (bimg::isDepth(bimg::TextureFormat::Enum(m_textureFormat) ) )
			{
				// eacp has no depth texture format (G4): a depth plane is part
				// of a colour render target. Made at createFrameBuffer, on the
				// colour texture the framebuffer pairs this with, or on a
				// colour target of this one's own (a depth-only framebuffer).
				// Sampled through RenderPass::setFragmentDepthTexture on that
				// target; always a 32-bit float plane, D24S8 and D0S8 with an
				// 8-bit stencil beside it. A depth array or cube keeps its
				// kind, which a depth-only framebuffer's carrier takes, so a
				// pass can attach it at a layer or face (ColorTarget::slice);
				// beside colour, the colour texture's own depth companion has
				// its layers. No API has a 3D depth buffer: a depth volume is
				// one 2D plane.
				if (GPU::TextureType::ThreeD == m_kind)
				{
					traceOnce(Unsupported::TextureLayers, "a 3D depth texture is created as one 2D plane (no API has a 3D depth buffer).");
					m_kind = GPU::TextureType::TwoD;
				}

				m_depthFormat   = true;
				m_stencilFormat = 0 < bimg::getBlockInfo(bimg::TextureFormat::Enum(m_textureFormat) ).stencilBits;
				m_numMips       = 1;
				m_depth         = 1;

				m_desc = GPU::TextureDescriptor();
				m_desc.width        = int(m_width);
				m_desc.height       = int(m_height);
				m_desc.format       = GPU::TextureFormat::RGBA8Unorm;
				m_desc.kind         = m_kind;
				m_desc.arrayLength  = GPU::TextureType::TwoDArray == m_kind || GPU::TextureType::CubeArray == m_kind ? int(m_numLayers) : 1;
				m_desc.renderTarget = true;
				m_desc.sampleCount  = GPU::TextureType::TwoD == m_kind ? sampleCountFor(_flags) : 1;
				return;
			}

			if (!toEacpFormat(m_textureFormat, m_format) )
			{
				traceOnce(Unsupported::UnknownFormat, "G4: a texture format with no eacp format and no BGRA8 decode is not created.");
				return;
			}

			if (0 != (_flags & BGFX_TEXTURE_SRGB) )
			{
				traceOnce(Unsupported::TextureSrgb, "G4: BGFX_TEXTURE_SRGB is sampled as linear (no sRGB formats in eacp).");
			}

			if (GPU::TextureType::CubeArray == m_kind
			&&  !GPU::Device::shared().supportsCubeArrays() )
			{
				traceOnce(Unsupported::TextureLayers, "a cube array on a device without cube arrays (Device::supportsCubeArrays) is not created.");
				return;
			}

			// BGFX_TEXTURE_BLIT_DST and BGFX_TEXTURE_READ_BACK ask nothing of eacp:
			// a blit is Frame::copy, which any texture takes, and every texture
			// reads back.

			GPU::TextureDescriptor desc;
			desc.width       = int(m_width);
			desc.height      = int(m_height);
			desc.format      = m_format;
			desc.kind        = m_kind;
			desc.arrayLength = GPU::TextureType::TwoDArray == m_kind || GPU::TextureType::CubeArray == m_kind ? int(m_numLayers) : 1;
			desc.volumeDepth = int(m_depth);

			// A render target of any kind: a framebuffer attachment names the
			// level and the layer, face or volume z a pass renders into
			// (ColorTarget::level, slice, layerCount). A kernel writes any
			// kind at any level, a cube or cube array as the 2D array of its
			// slices, which is how bgfx's kernels declare one (IMAGE2D_ARRAY,
			// bound through setOutputTextureAsArray), in any format the device
			// stores to.
			desc.renderTarget = renderTarget;
			desc.computeWrite = computeWrite && GPU::Device::shared().supportsComputeWrite(m_format);

			if (computeWrite
			&&  !desc.computeWrite)
			{
				traceOnce(Unsupported::ComputeWriteFormat, "a compute-write texture in a format the device cannot compute-write (Device::supportsComputeWrite: BGRA8, block formats) is created read-only.");
			}

			if (desc.renderTarget
			&&  GPU::TextureType::TwoD == m_kind)
			{
				desc.sampleCount = sampleCountFor(_flags);

				if (1 < desc.sampleCount)
				{
					// A multisampled target has the one level.
					m_numMips = 1;
				}
			}

			// The caller's chain (on a render target, filled by passes and
			// kernels at a level, or built from level 0 when a framebuffer
			// attachment asks for BGFX_ATTACHMENT_AUTO_GEN_MIPS), whatever the kind (a cube's per face).
			desc.mipLevels = 1 < m_numMips ? int(m_numMips) : 0;

			// Every level of every slice the container holds, in the format eacp
			// stores: straight out of bgfx's memory where the bytes are eacp's,
			// decoded into a buffer of its own where not. A texture
			// made with no data (a target, a dynamic texture) names none.
			const bool convert = m_textureFormat != m_requestedFormat;
			const uint32_t numSlices = slices();

			std::vector<GPU::TextureSubresource> subresources;
			std::vector<std::vector<uint8_t> > storage;
			std::vector<uint8_t> temp;

			subresources.reserve(numSlices*m_numMips);
			storage.reserve(convert ? numSlices*m_numMips : 0);

			for (uint32_t slice = 0; slice < numSlices; ++slice)
			{
				for (uint8_t lod = 0; lod < m_numMips; ++lod)
				{
					bimg::ImageMip mip;

					if (!bimg::imageGetRawData(imageContainer, uint16_t(slice), lod+startLod, _mem->data, _mem->size, mip) )
					{
						continue;
					}

					const uint32_t width  = mipExtent(m_width,  lod);
					const uint32_t height = mipExtent(m_height, lod);
					const uint32_t depth  = isVolume() ? mipExtent(m_depth, lod) : 1;

					GPU::TextureSubresource sub;
					sub.level = int(lod);
					sub.slice = int(slice);

					if (convert)
					{
						// Decoded image by image at bimg's block-aligned size,
						// then the level's own rows are kept.
						const uint32_t srcImage = mip.m_size / bx::max<uint32_t>(1, mip.m_depth);

						storage.emplace_back(width*height*depth*4, 0);
						uint8_t* dst = storage.back().data();
						temp.resize(mip.m_width*mip.m_height*4);

						for (uint32_t zz = 0; zz < depth && zz < bx::max<uint32_t>(1, mip.m_depth); ++zz)
						{
							bimg::imageDecodeToBgra8(g_allocator, temp.data(), mip.m_data + zz*srcImage, mip.m_width, mip.m_height, mip.m_width*4, mip.m_format);

							for (uint32_t yy = 0; yy < height && yy < mip.m_height; ++yy)
							{
								bx::memCopy(&dst[(zz*height + yy)*width*4], &temp[yy*mip.m_width*4], bx::min(width, mip.m_width)*4);
							}
						}

						sub.pixels = dst;
					}
					else
					{
						if (mip.m_size < levelBytes(lod)*depth)
						{
							continue;
						}

						sub.pixels = mip.m_data;
					}

					subresources.push_back(sub);
				}
			}

			m_desc = desc;
			m_texture.emplace(GPU::Device::shared().makeTextureFromSubresources(desc, subresources) );

			if (!m_texture->isValid() )
			{
				BX_TRACE("eacp: makeTextureFromSubresources refused %dx%dx%d %s, %d layers%s, %d mips."
					, m_width, m_height, m_depth, getName(m_textureFormat), m_numLayers, m_cube ? " x6" : "", m_numMips);
				m_texture.reset();
				return;
			}

			m_numMips = uint8_t(m_texture->mipLevels() );
		}

		void destroy()
		{
			m_texture.reset();
			m_depthFormat   = false;
			m_stencilFormat = false;
			m_depthKept     = false;
			m_depthOf       = kInvalidHandle;
		}

		/// One box of one level of one slice, through updateSubresource: bgfx's
		/// _z is the layer of an array or cube array (_side the face), or the
		/// first image of a volume, _depth the images.
		void update(uint8_t _side, uint8_t _mip, const Rect& _rect, uint16_t _z, uint16_t _depth, uint16_t _pitch, const Memory* _mem)
		{
			if (!m_texture)
			{
				return;
			}

			const uint32_t mipWidth  = mipExtent(m_width,  _mip);
			const uint32_t mipHeight = mipExtent(m_height, _mip);
			const uint32_t mipDepth  = isVolume() ? mipExtent(m_depth, _mip) : 1;
			const uint32_t depth     = isVolume() ? bx::max<uint32_t>(1, _depth) : 1;
			const uint32_t z         = isVolume() ? _z : 0;
			const uint32_t slice     = isVolume() ? 0 : uint32_t(_z)*m_numSides + _side;

			// A level of a block format smaller than a block is updated with a
			// whole block, which bgfx names as a block-sized rect (stored or
			// decoded); the region eacp takes ends at the level's edge.
			const bimg::ImageBlockInfo& requestedBlock = bimg::getBlockInfo(bimg::TextureFormat::Enum(m_requestedFormat) );
			const bool     blocks    = GPU::isCompressedFormat(m_format);
			const uint32_t boundW    = bx::strideAlign(mipWidth,  requestedBlock.blockWidth);
			const uint32_t boundH    = bx::strideAlign(mipHeight, requestedBlock.blockHeight);

			if (_mip >= m_numMips
			||  _side >= m_numSides
			||  slice >= slices()
			||  z + depth > mipDepth
			||  _rect.m_x + _rect.m_width  > boundW
			||  _rect.m_y + _rect.m_height > boundH
			||  _rect.m_x >= mipWidth
			||  _rect.m_y >= mipHeight
			||  0 == _rect.m_width
			||  0 == _rect.m_height)
			{
				traceOnce(Unsupported::TextureUpdate, "an update outside the texture's levels, slices or extent is skipped.");
				return;
			}

			const bimg::TextureFormat::Enum requested = bimg::TextureFormat::Enum(m_requestedFormat);
			const bimg::ImageBlockInfo& blockInfo = bimg::getBlockInfo(requested);
			const bool compressed = bimg::isCompressed(requested);

			const uint32_t tightPitch = compressed
				? ( (_rect.m_width + blockInfo.blockWidth - 1)/blockInfo.blockWidth)*blockInfo.blockSize
				: _rect.m_width*blockInfo.bitsPerPixel/8
				;
			const uint32_t srcPitch = UINT16_MAX == _pitch ? tightPitch : _pitch;
			const uint32_t rows     = compressed
				? (_rect.m_height + blockInfo.blockHeight - 1)/blockInfo.blockHeight
				: _rect.m_height
				;
			const uint32_t srcImage = srcPitch*rows;

			if ( (depth-1)*srcImage + srcPitch*(rows-1) + tightPitch > _mem->size)
			{
				return;
			}

			const uint8_t* data  = _mem->data;
			uint32_t       pitch = srcPitch;
			std::vector<uint8_t> temp;

			if (m_textureFormat != m_requestedFormat)
			{
				// bimg decodes and converts from tight rows.
				std::vector<uint8_t> tight(tightPitch*rows*depth);
				for (uint32_t zz = 0; zz < depth; ++zz)
				{
					for (uint32_t yy = 0; yy < rows; ++yy)
					{
						bx::memCopy(&tight[(zz*rows + yy)*tightPitch], &data[zz*srcImage + yy*srcPitch], tightPitch);
					}
				}

				const uint32_t imageBytes = tightPitch*rows;

				pitch = _rect.m_width*4;
				temp.resize(pitch*_rect.m_height*depth);

				for (uint32_t zz = 0; zz < depth; ++zz)
				{
					bimg::imageDecodeToBgra8(g_allocator, &temp[zz*pitch*_rect.m_height], &tight[zz*imageBytes], _rect.m_width, _rect.m_height, pitch, requested);
				}

				data = temp.data();
			}

			const uint32_t storedRows = blocks ? rows : _rect.m_height;

			GPU::TextureRegion region;
			region.level         = int(_mip);
			region.slice         = int(slice);
			region.origin        = { int(_rect.m_x), int(_rect.m_y), int(z) };
			region.size          = { int(bx::min<uint32_t>(_rect.m_width, mipWidth - _rect.m_x) ), int(bx::min<uint32_t>(_rect.m_height, mipHeight - _rect.m_y) ), int(depth) };
			region.bytesPerRow   = int(pitch);
			region.bytesPerImage = int(pitch*storedRows);

			m_texture->updateSubresource(region, data);
		}

		/// One level of one slice as bgfx lays it out (the first image of a
		/// volume's level, as renderer_mtl.cpp reads).
		void read(void* _data, uint16_t _layer, uint8_t _mip)
		{
			if (!m_texture)
			{
				return;
			}

			if (_mip >= m_numMips
			||  _layer >= slices() )
			{
				traceOnce(Unsupported::TextureRead, "readTexture of a level or layer the texture does not have is not read.");
				return;
			}

			if (GPU::isCompressedFormat(m_format) )
			{
				traceOnce(Unsupported::TextureRead, "readTexture of a block-compressed texture is not read (eacp's readSubresource refuses blocks).");
				return;
			}

			const uint32_t width  = mipExtent(m_width,  _mip);
			const uint32_t height = mipExtent(m_height, _mip);

			GPU::TextureRegion region;
			region.level  = int(_mip);
			region.slice  = int(_layer);
			region.origin = { 0, 0, 0 };
			region.size   = { int(width), int(height), 1 };

			m_texture->readSubresource(region, _data);
		}

		std::optional<GPU::Texture> m_texture;
		uint64_t m_flags;
		uint32_t m_width;
		uint32_t m_height;
		uint32_t m_depth;     // a volume's images, 1 otherwise
		uint16_t m_numLayers; // an array's layers (cubes on a cube array), 1 otherwise
		uint8_t  m_numMips;
		uint8_t  m_numSides;
		TextureFormat::Enum m_requestedFormat;
		TextureFormat::Enum m_textureFormat;
		GPU::TextureFormat  m_format;
		GPU::TextureType    m_kind;
		GPU::TextureDescriptor m_desc;
		bool m_cube;
		bool m_depthFormat;   // a bgfx depth (or stencil) format, no eacp texture of its own
		bool m_stencilFormat; // ... with a stencil plane
		bool m_depthKept;     // the depth plane holds what a pass kept (DepthAction::Resume may load it)
		uint16_t m_depthOf;   // a depth texture: the texture whose target carries its plane (itself, or a multisampled colour 0)
	};

	/// A bgfx framebuffer of textures: its colour attachments in order, and
	/// its depth. The pass renders into the colour textures, attachment n of
	/// the pass being m_color[n], or for a depth-only framebuffer into the
	/// depth texture's own carrier target; m_target is the texture the pass
	/// is opened on (colour 0, or that carrier). The depth plane is the depth
	/// texture's own carrier's, attached through RenderPassDescriptor::
	/// depthTarget at the depth attachment's image (m_ownDepth), or, for a
	/// multisampled framebuffer (depthTarget is single-sampled only, G88),
	/// colour 0's own plane at colour 0's image (G36).
	///
	/// Each attachment names the image it renders into: bgfx's mip is eacp's
	/// level, its layer the slice (an array layer, a cube face, layer*6 + face
	/// on a cube array, a volume's z), its numLayers the run of slices a
	/// layered pass covers (ColorTarget::level, slice, layerCount). m_sub[n]
	/// is colour attachment n's; m_depthSub the depth attachment's, which is
	/// what a depth-only framebuffer's carrier renders at.
	struct FrameBufferEacp
	{
		struct Sub
		{
			uint8_t  m_level  = 0;
			uint16_t m_slice  = 0;
			uint16_t m_layers = 1;

			bool isDefault() const
			{
				return 0 == m_level && 0 == m_slice && 1 == m_layers;
			}

			bool operator==(const Sub& _other) const
			{
				return m_level == _other.m_level && m_slice == _other.m_slice && m_layers == _other.m_layers;
			}
		};

		uint16_t m_color[BGFX_CONFIG_MAX_FRAME_BUFFER_ATTACHMENTS];
		Sub      m_sub[BGFX_CONFIG_MAX_FRAME_BUFFER_ATTACHMENTS];
		Sub      m_depthSub;

		/// BGFX_ATTACHMENT_AUTO_GEN_MIPS on colour attachment n: its texture's
		/// levels below 0 are built from level 0 (Frame::generateMipmaps)
		/// when the views on this framebuffer end, as renderer_mtl.cpp's
		/// FrameBufferMtl::resolve does.
		bool     m_autoGen[BGFX_CONFIG_MAX_FRAME_BUFFER_ATTACHMENTS] = {};
		uint8_t  m_num      = 0;
		uint16_t m_depth    = kInvalidHandle;
		uint16_t m_target   = kInvalidHandle;
		bool     m_ownDepth = false;
		bool     m_valid    = false;

		/// Whether a pass over it goes through ColorTarget (several colour
		/// attachments, or any image other than level 0, slice 0).
		bool needsTargets() const
		{
			if (1 < m_num)
			{
				return true;
			}

			return 0 == m_num
				? !m_depthSub.isDefault()
				: !m_sub[0].isDefault()
				;
		}

		/// The image a pass renders into: colour attachment 0's, or a
		/// depth-only framebuffer's depth attachment's.
		const Sub& targetSub() const
		{
			return 0 == m_num ? m_depthSub : m_sub[0];
		}
	};

	struct PipelineEacp
	{
		std::optional<GPU::RenderPipeline> m_pipeline;
		uint16_t m_program;
	};

	/// Makes a pass from inside std::optional::emplace: RenderPass can be neither
	/// copied nor moved, and Frame::beginPass returns one by value. The window
	/// with no texture, one texture, or several colour targets each with its
	/// own clear.
	struct PassMaker
	{
		operator GPU::RenderPass() const
		{
			if (!m_targets.empty() )
			{
				return m_frame.beginMultiTargetPass(m_targets, m_descriptor);
			}

			return NULL == m_texture
				? m_frame.beginPass(m_descriptor)
				: m_frame.beginPass(*m_texture, m_descriptor)
				;
		}

		GPU::Frame& m_frame;
		const GPU::RenderPassDescriptor& m_descriptor;
		const GPU::Texture* m_texture;
		std::span<const GPU::ColorTarget> m_targets = {};
	};

	/// The same for Frame::beginCompute: ComputePass can be neither copied nor
	/// moved either. Serial order, so each dispatch sees the writes of the
	/// ones before it, as bgfx's compute items in sort order expect.
	struct ComputePassMaker
	{
		operator GPU::ComputePass() const
		{
			return m_frame.beginCompute(m_label);
		}

		GPU::Frame& m_frame;
		const char* m_label;
	};

	/// A vertex buffer slot of the open pass, by the Metal buffer (or its
	/// counterpart) and offset it was last bound to.
	struct BoundStream
	{
		/// False when _range is what the slot already holds.
		bool set(const GPU::BufferRange& _range)
		{
			const void* native = NULL != _range.buffer ? _range.buffer->nativeBuffer() : NULL;

			if (native == m_native
			&&  _range.offset == m_offset)
			{
				return false;
			}

			m_native = native;
			m_offset = _range.offset;
			return true;
		}

		const void* m_native = NULL;
		int64_t     m_offset = -1;
	};

	struct RendererContextEacp : public RendererContextI
	{
		RendererContextEacp()
			: m_view(NULL)
			, m_vertexStream(GPU::BufferUsage::Vertex)
			, m_indexStream(GPU::BufferUsage::Index)
		{
			bx::memSet(m_indexBuffers,  0, sizeof(m_indexBuffers) );
			bx::memSet(m_vertexBuffers, 0, sizeof(m_vertexBuffers) );
			bx::memSet(m_shaders,       0, sizeof(m_shaders) );
			bx::memSet(m_program,       0, sizeof(m_program) );
		}

		~RendererContextEacp()
		{
			m_computePass.reset();
			m_pass.reset();

			for (auto& pipeline : m_pipelines)
			{
				bx::deleteObject(g_allocator, pipeline.second);
			}

			destroyAll(m_program);
			destroyAll(m_shaders);
			destroyAll(m_vertexBuffers);
			destroyAll(m_indexBuffers);
		}

		template<typename Ty, size_t NumT>
		static void destroyAll(Ty* (&_array)[NumT])
		{
			for (size_t ii = 0; ii < NumT; ++ii)
			{
				if (NULL != _array[ii])
				{
					bx::deleteObject(g_allocator, _array[ii]);
					_array[ii] = NULL;
				}
			}
		}

		template<typename Ty, size_t NumT>
		static void destroyOne(Ty* (&_array)[NumT], uint16_t _idx)
		{
			if (NULL != _array[_idx])
			{
				bx::deleteObject(g_allocator, _array[_idx]);
				_array[_idx] = NULL;
			}
		}

		bool init(const Init& _init)
		{
			BX_ASSERT(::eacp::Threads::isMainThread()
				, "The eacp renderer runs on eacp's main thread: call bgfx::renderFrame from there."
				);

			GPU::Device& device = GPU::Device::shared();

			if (!device.isValid() )
			{
				BX_TRACE("eacp: Device::shared() is not valid.");
				return false;
			}

			m_view = static_cast<GPU::GPUView*>(_init.swapChain.nwh);
			s_hiDpi = 0 != (_init.swapChain.flags & BGFX_SWAP_CHAIN_HIDPI);

			g_caps.vendorId = BGFX_PCI_ID_NONE;
			g_caps.deviceId = 0;
			g_caps.numGPUs  = 1;
			g_caps.gpu[0].vendorId = g_caps.vendorId;
			g_caps.gpu[0].deviceId = g_caps.deviceId;

			g_caps.homogeneousDepth = false;
			g_caps.originBottomLeft = false;

			// IMAGE_RW: a compute-write texture is ShaderRead|ShaderWrite, and
			// the device reads and writes some format through one bind
			// (supportsComputeReadWrite, R32Float on every read-write tier);
			// which formats is each format's IMAGE_READ (fillFormatCaps).
			// BLEND_INDEPENDENT: a pipeline blends each of a pass's colour
			// attachments by its own ColorTargetState (G3).
			// DRAW_INDIRECT: RenderPass::drawIndirect and drawIndexedIndirect
			// read bgfx's records, and ComputePass::dispatchIndirect its
			// dispatches (one bit for both). DRAW_INDIRECT_COUNT only where the
			// device draws a count read from a buffer, which Metal does not;
			// bgfx asserts on the count form of submit without it, and in a
			// build without asserts eacp refuses the draw with a trace.
			g_caps.supported |= 0
				| BGFX_CAPS_COMPUTE
				| BGFX_CAPS_INDEX32
				| (device.supportsComputeReadWrite(GPU::TextureFormat::R32Float) ? BGFX_CAPS_IMAGE_RW : 0)
				| BGFX_CAPS_BLEND_INDEPENDENT
				| BGFX_CAPS_DRAW_INDIRECT
				| (device.supportsIndirectCount() ? BGFX_CAPS_DRAW_INDIRECT_COUNT : 0)
				// 2D arrays are maxTextureLayers above 1 and volumes
				// BGFX_CAPS_FORMAT_TEXTURE_3D (fillFormatCaps); bgfx has no
				// caps bit for either.
				| (device.supportsCubeArrays() ? BGFX_CAPS_TEXTURE_CUBE_ARRAY : 0)
				// A framebuffer attachment of several layers is one layered
				// pass (ColorTarget::layerCount), the vertex shader choosing
				// each primitive's with gl_Layer, which eacp-metal compiles to
				// [[render_target_array_index]].
				| (device.supportsLayeredRendering() ? BGFX_CAPS_VIEWPORT_LAYER_ARRAY : 0)
				;

			fillFormatCaps(device);
			m_vertexTextures = device.supportsVertexTextures();

			createFallbackTextures();

			// eacp reports no limit; 16384 is what every Metal GPU since the
			// A8/Mac2 family takes, and D3D12 feature level 11's floor.
			g_caps.limits.maxTextureSize      = 16384;
			// Layers of a 2D array (Device::maxTextureLayers, 2048 on all three
			// backends); a cube array's limit is cubes times six within it,
			// which bgfx has no limit of its own for. bgfx has no 3D size limit
			// either: eacp refuses a volume past maxTextureSize3D (2048).
			g_caps.limits.maxTextureLayers    = uint32_t(bx::max(device.maxTextureLayers(), 1) );
			g_caps.limits.maxFBAttachments    = bx::min<uint32_t>(GPU::maxColorAttachments, BGFX_CONFIG_MAX_FRAME_BUFFER_ATTACHMENTS);
			// A stage's texture slots, each with its per-bind sampler: 16 on
			// Metal and Vulkan, 8 on D3D12 (Device::maxTextureSlots).
			g_caps.limits.maxTextureSamplers  = bx::min<uint32_t>(BGFX_CONFIG_MAX_TEXTURE_SAMPLERS, uint32_t(bx::max(device.maxTextureSlots(), 0) ) );
			// The device's own storage-buffer slots for a kernel (16 on Metal, 8
			// on D3D12), within bgfx's bind array; a compute stage's texture or
			// image binds at texture(stage), which Metal has as many of.
			g_caps.limits.maxComputeBindings  = bx::min<uint32_t>(BGFX_MAX_COMPUTE_BINDINGS, uint32_t(bx::max(device.maxComputeBufferSlots(), 0) ) );
			g_caps.limits.maxVertexStreams    = BGFX_CONFIG_MAX_VERTEX_STREAMS;
			g_caps.limits.maxVertexAttributes = 16;

			// A copy between a buffer and a texture takes a buffer offset on
			// four bytes and the texel size, and rows of whole texels
			// (Frame::copy); 16 is every eacp texel size's multiple.
			g_caps.limits.blitOffsetAlign     = 16;
			g_caps.limits.blitRowPitchAlign   = 1;

			m_perfTrace = NULL != getenv("EACP_BGFX_CAPTURE");

			BX_TRACE("eacp: init on \"%s\", view %p.", device.name().c_str(), m_view);

			return true;
		}

		static void fillFormatCaps(GPU::Device& _device)
		{
			for (uint32_t ii = 0; ii < TextureFormat::Count; ++ii)
			{
				g_caps.formats[ii] = BGFX_CAPS_FORMAT_TEXTURE_NONE;
			}

			const uint32_t renderable = 0
				| BGFX_CAPS_FORMAT_TEXTURE_2D
				| BGFX_CAPS_FORMAT_TEXTURE_FRAMEBUFFER
				| BGFX_CAPS_FORMAT_TEXTURE_MSAA
				| BGFX_CAPS_FORMAT_TEXTURE_FRAMEBUFFER_MSAA
				;

			// What toEacpFormat maps: 2D, cube and (uncompressed) 3D, each also
			// as an array where bgfx asks for layers (maxTextureLayers,
			// BGFX_CAPS_TEXTURE_CUBE_ARRAY), renderable where eacp renders to it. No _SRGB (G4),
			// _VERTEX where the device samples in the vertex stage (G17),
			// _MIP_AUTOGEN where Frame::generateMipmaps filters the format
			// (Device::supportsMipGeneration), _IMAGE_* as the device answers for kernels
			// (Device::supportsComputeWrite, supportsComputeReadWrite). _MSAA and _FRAMEBUFFER_MSAA
			// on what renders, depth included: eacp multisamples a render target
			// and resolves it (sampleCountFor picks the count the device takes).
			// bgfx adds _EMULATED to every other format bimg can
			// decode to BGRA8, and createTexture does that decode.
			const uint32_t vertex = _device.supportsVertexTextures() ? BGFX_CAPS_FORMAT_TEXTURE_VERTEX : 0;

			const uint32_t sampled = 0
				| BGFX_CAPS_FORMAT_TEXTURE_2D
				| BGFX_CAPS_FORMAT_TEXTURE_CUBE
				| BGFX_CAPS_FORMAT_TEXTURE_3D
				| vertex
				;

			// eacp refuses a compressed volume.
			const uint32_t blocks = sampled & ~BGFX_CAPS_FORMAT_TEXTURE_3D;

			// Every uncompressed format eacp has renders, each in its own
			// PixelFormat; R32U as an integer attachment, never blended (and
			// so with no _MSAA resolve to average).
			static const TextureFormat::Enum s_uncompressed[] =
			{
				TextureFormat::BGRA8,
				TextureFormat::RGBA8,
				TextureFormat::R8,
				TextureFormat::RG8,
				TextureFormat::RGBA16F,
				TextureFormat::RGBA32F,
				TextureFormat::R32F,
				TextureFormat::R16F,
				TextureFormat::RG16F,
				TextureFormat::RG32F,
				TextureFormat::R16,
				TextureFormat::R32U,
			};

			for (TextureFormat::Enum format : s_uncompressed)
			{
				GPU::TextureFormat eacp;
				toEacpFormat(format, eacp);

				// IMAGE_WRITE where a kernel can store to the format
				// (Device::supportsComputeWrite: on Metal every uncompressed
				// format but BGRA8). IMAGE_READ where it can also load through
				// the bind it writes with (supportsComputeReadWrite: Metal's
				// read-write tier), so READ|WRITE, the pair bgfx asks for an
				// IMAGE2D_RW, is a format an access::read_write texture takes;
				// an IMAGE2D_RO of any writable format reads either way.
				g_caps.formats[format] = renderable | sampled
					| (_device.supportsComputeWrite(eacp)     ? BGFX_CAPS_FORMAT_TEXTURE_IMAGE_WRITE : 0)
					| (_device.supportsComputeReadWrite(eacp) ? BGFX_CAPS_FORMAT_TEXTURE_IMAGE_READ  : 0)
					| (_device.supportsMipGeneration(eacp)    ? BGFX_CAPS_FORMAT_TEXTURE_MIP_AUTOGEN : 0)
					;

				if (GPU::isIntegerFormat(eacp) )
				{
					g_caps.formats[format] &= ~(BGFX_CAPS_FORMAT_TEXTURE_MSAA|BGFX_CAPS_FORMAT_TEXTURE_FRAMEBUFFER_MSAA);
				}
			}

			g_caps.formats[TextureFormat::BGRA8] |= BGFX_CAPS_FORMAT_TEXTURE_BACKBUFFER;

			// Depth: a 32-bit float plane (with 8 bits of stencil for the
			// formats that have it) on a colour target, rendered into by a
			// framebuffer and sampled through setFragmentDepthTexture, with or
			// without a compare sampler. Every depth format is that one plane,
			// so D16 and D24 are stored at more precision than asked.
			const uint32_t depth = 0
				| BGFX_CAPS_FORMAT_TEXTURE_2D
				| vertex
				| BGFX_CAPS_FORMAT_TEXTURE_FRAMEBUFFER
				| BGFX_CAPS_FORMAT_TEXTURE_MSAA
				| BGFX_CAPS_FORMAT_TEXTURE_FRAMEBUFFER_MSAA
				;

			g_caps.formats[TextureFormat::D16  ] = depth;
			g_caps.formats[TextureFormat::D24  ] = depth;
			g_caps.formats[TextureFormat::D24S8] = depth;
			g_caps.formats[TextureFormat::D32  ] = depth;
			g_caps.formats[TextureFormat::D16F ] = depth;
			g_caps.formats[TextureFormat::D24F ] = depth;
			g_caps.formats[TextureFormat::D32F ] = depth;
			g_caps.formats[TextureFormat::D0S8 ] = BGFX_CAPS_FORMAT_TEXTURE_FRAMEBUFFER;

			if (_device.supportsBlockCompression() )
			{
				g_caps.formats[TextureFormat::BC1] = blocks;
				g_caps.formats[TextureFormat::BC2] = blocks;
				g_caps.formats[TextureFormat::BC3] = blocks;
				g_caps.formats[TextureFormat::BC7] = blocks;
			}
		}

		/// What a sampler slot reads when the draw binds nothing there, or a
		/// texture eacp could not create (or of another kind than the shader
		/// declares): opaque white, one per TextureType, 1x1 (x1).
		void createFallbackTextures()
		{
			const uint8_t white[6*4] =
			{
				0xff, 0xff, 0xff, 0xff,  0xff, 0xff, 0xff, 0xff,  0xff, 0xff, 0xff, 0xff,
				0xff, 0xff, 0xff, 0xff,  0xff, 0xff, 0xff, 0xff,  0xff, 0xff, 0xff, 0xff,
			};

			for (uint32_t ii = 0; ii < BX_COUNTOF(m_fallback); ++ii)
			{
				TextureEacp& fallback = m_fallback[ii];
				const GPU::TextureType kind = GPU::TextureType(ii);
				const bool cube = GPU::TextureType::Cube == kind || GPU::TextureType::CubeArray == kind;

				if (GPU::TextureType::CubeArray == kind
				&&  !GPU::Device::shared().supportsCubeArrays() )
				{
					continue;
				}

				GPU::TextureDescriptor desc;
				desc.width  = 1;
				desc.height = 1;
				desc.format = GPU::TextureFormat::RGBA8Unorm;
				desc.kind   = kind;

				fallback.m_texture.emplace(GPU::Device::shared().makeTexture(desc, white) );
				fallback.m_kind            = kind;
				fallback.m_cube            = cube;
				fallback.m_numSides        = cube ? 6 : 1;
				fallback.m_width           = 1;
				fallback.m_height          = 1;
				fallback.m_flags           = BGFX_SAMPLER_POINT;
				fallback.m_requestedFormat = TextureFormat::RGBA8;
				fallback.m_textureFormat   = TextureFormat::RGBA8;
			}

			// A sampleable depth plane at the far plane, for a depth texture
			// with nothing rendered into it yet: the first submit clears it.
			GPU::TextureDescriptor desc;
			desc.width        = 1;
			desc.height       = 1;
			desc.format       = GPU::TextureFormat::RGBA8Unorm;
			desc.renderTarget = true;
			m_fallbackDepth.m_desc   = desc;
			m_fallbackDepth.m_width  = 1;
			m_fallbackDepth.m_height = 1;
			m_fallbackDepth.makeTarget(true, false, true);
		}

		RendererType::Enum getRendererType() const override
		{
			return RendererType::Eacp;
		}

		const char* getRendererName() const override
		{
			return BGFX_RENDERER_EACP_NAME;
		}

		bool isDeviceRemoved() override
		{
			return false;
		}

		void flip() override
		{
		}

		void createIndexBuffer(IndexBufferHandle _handle, const Memory* _mem, uint16_t _flags) override
		{
			destroyOne(m_indexBuffers, _handle.idx);
			m_indexBuffers[_handle.idx] = BX_NEW(g_allocator, BufferEacp)(_mem->size, _mem->data, _flags, false, VertexLayoutHandle(BGFX_INVALID_HANDLE) );
		}

		void destroyIndexBuffer(IndexBufferHandle _handle) override
		{
			destroyOne(m_indexBuffers, _handle.idx);
		}

		void createVertexLayout(VertexLayoutHandle _handle, const VertexLayout& _layout) override
		{
			bx::memCopy(&m_vertexLayouts[_handle.idx], &_layout, sizeof(VertexLayout) );
		}

		void destroyVertexLayout(VertexLayoutHandle /*_handle*/) override
		{
		}

		void createVertexBuffer(VertexBufferHandle _handle, const Memory* _mem, VertexLayoutHandle _layoutHandle, uint16_t _flags) override
		{
			destroyOne(m_vertexBuffers, _handle.idx);
			m_vertexBuffers[_handle.idx] = BX_NEW(g_allocator, BufferEacp)(_mem->size, _mem->data, _flags, true, _layoutHandle);
		}

		void destroyVertexBuffer(VertexBufferHandle _handle) override
		{
			destroyOne(m_vertexBuffers, _handle.idx);
		}

		void createDynamicIndexBuffer(IndexBufferHandle _handle, uint32_t _size, uint16_t _flags) override
		{
			destroyOne(m_indexBuffers, _handle.idx);
			m_indexBuffers[_handle.idx] = BX_NEW(g_allocator, BufferEacp)(_size, NULL, _flags, false, VertexLayoutHandle(BGFX_INVALID_HANDLE) );
		}

		void updateDynamicIndexBuffer(IndexBufferHandle _handle, uint32_t _offset, uint32_t _size, const Memory* _mem) override
		{
			if (NULL != m_indexBuffers[_handle.idx])
			{
				m_indexBuffers[_handle.idx]->update(_offset, bx::min(_size, _mem->size), _mem->data);
			}
		}

		void destroyDynamicIndexBuffer(IndexBufferHandle _handle) override
		{
			destroyOne(m_indexBuffers, _handle.idx);
		}

		void createDynamicVertexBuffer(VertexBufferHandle _handle, uint32_t _size, uint16_t _flags) override
		{
			destroyOne(m_vertexBuffers, _handle.idx);
			m_vertexBuffers[_handle.idx] = BX_NEW(g_allocator, BufferEacp)(_size, NULL, _flags, true, VertexLayoutHandle(BGFX_INVALID_HANDLE) );
		}

		void updateDynamicVertexBuffer(VertexBufferHandle _handle, uint32_t _offset, uint32_t _size, const Memory* _mem) override
		{
			if (NULL != m_vertexBuffers[_handle.idx])
			{
				m_vertexBuffers[_handle.idx]->update(_offset, bx::min(_size, _mem->size), _mem->data);
			}
		}

		void destroyDynamicVertexBuffer(VertexBufferHandle _handle) override
		{
			destroyOne(m_vertexBuffers, _handle.idx);
		}

		void createShader(ShaderHandle _handle, const Memory* _mem) override
		{
			destroyOne(m_shaders, _handle.idx);

			ShaderEacp* shader = BX_NEW(g_allocator, ShaderEacp);

			if (!shader->create(_mem, m_uniformReg) )
			{
				bx::deleteObject(g_allocator, shader);
				return;
			}

			m_shaders[_handle.idx] = shader;
		}

		void destroyShader(ShaderHandle _handle) override
		{
			destroyOne(m_shaders, _handle.idx);
		}

		void createProgram(ProgramHandle _handle, ShaderHandle _vsh, ShaderHandle _fsh) override
		{
			destroyProgram(_handle);

			const ShaderEacp* vsh = m_shaders[_vsh.idx];
			const ShaderEacp* fsh = isValid(_fsh) ? m_shaders[_fsh.idx] : NULL;

			if (NULL == vsh)
			{
				return;
			}

			ProgramEacp* program = BX_NEW(g_allocator, ProgramEacp);
			program->create(vsh, fsh);
			m_program[_handle.idx] = program;
		}

		void destroyProgram(ProgramHandle _handle) override
		{
			if (NULL == m_program[_handle.idx])
			{
				return;
			}

			for (auto it = m_pipelines.begin(); it != m_pipelines.end(); )
			{
				if (it->second->m_program == _handle.idx)
				{
					bx::deleteObject(g_allocator, it->second);
					it = m_pipelines.erase(it);
				}
				else
				{
					++it;
				}
			}

			m_lastPipelineKey = 0;
			m_lastPipeline    = NULL;

			destroyOne(m_program, _handle.idx);
		}

		void* createTexture(TextureHandle _handle, const Memory* _mem, uint64_t _flags, uint8_t _skip, uint64_t _external) override
		{
			if (0 != _external)
			{
				traceOnce(Unsupported::TextureExternal, "an external native texture is ignored; bgfx's data is used.");
			}

			m_textures[_handle.idx].create(_mem, _flags, _skip);
			return NULL;
		}

		void updateTexture(TextureHandle _handle, uint8_t _side, uint8_t _mip, const Rect& _rect, uint16_t _z, uint16_t _depth, uint16_t _pitch, const Memory* _mem) override
		{
			m_textures[_handle.idx].update(_side, _mip, _rect, _z, _depth, _pitch, _mem);
		}

		void clearTexture(TextureHandle /*_handle*/, uint8_t /*_mip*/, uint8_t /*_numMips*/, uint16_t /*_layer*/, uint16_t /*_numLayers*/) override
		{
			traceOnce(Unsupported::ClearTexture, "clearTexture is not done (no eacp texture clear; a render pass would do it in step 5).");
		}

		/// Blocking: bgfx reports the data ready two frames on, by which time
		/// it already is.
		void readTexture(TextureHandle _handle, void* _data, uint16_t _layer, uint8_t _mip) override
		{
			TextureEacp& texture = m_textures[_handle.idx];

			if (texture.m_depthFormat)
			{
				traceOnce(Unsupported::TextureRead, "readTexture of a depth texture is not read (eacp reads colour only).");
				return;
			}

			if (texture.isTarget()
			||  0 != (texture.m_flags & BGFX_TEXTURE_BLIT_DST) )
			{
				// What the GPU drew or copied: the frame so far is committed
				// first.
				flushFrame();
			}

			texture.read(_data, _layer, _mip);
		}

		/// Buffer::read of a compute or indirect buffer (bgfx refuses the rest),
		/// valid once the work that wrote it is committed: inside the frame,
		/// the frame so far is flushed first, as for readTexture.
		void readBuffer(Handle _handle, void* _data, uint32_t _offset, uint32_t _size) override
		{
			BufferEacp* buffer = bufferOf(_handle);

			if (NULL == buffer
			||  !buffer->m_static)
			{
				return;
			}

			flushFrame();
			buffer->m_static->read(_data, _size, _offset);
		}

		/// As renderer_mtl.cpp: a texture of the new size, recreated from a
		/// TextureCreate header with no data.
		void resizeTexture(TextureHandle _handle, uint16_t _width, uint16_t _height, uint8_t _numMips, uint16_t _numLayers) override
		{
			TextureEacp& texture = m_textures[_handle.idx];

			const Memory* mem = alloc(sizeof(uint32_t) + sizeof(TextureCreate) );

			bx::StaticMemoryBlockWriter writer(mem->data, mem->size);
			bx::write(&writer, kChunkMagicTex, bx::ErrorAssert{});

			TextureCreate tc;
			tc.m_width     = _width;
			tc.m_height    = _height;
			tc.m_depth     = 0;
			tc.m_numLayers = _numLayers;
			tc.m_numMips   = _numMips;
			tc.m_format    = texture.m_requestedFormat;
			tc.m_cubeMap   = false;
			tc.m_mem       = NULL;
			bx::write(&writer, tc, bx::ErrorAssert{});

			// A depth plane made by createFrameBuffer goes with the texture to
			// its new size.
			const bool depth      = texture.m_texture && texture.m_texture->hasDepth();
			const bool stencil    = texture.m_texture && texture.m_texture->hasStencil();
			const bool sampleable = texture.m_texture && texture.m_texture->hasSampleableDepth();
			const uint16_t depthOf = texture.m_depthOf;

			texture.create(mem, texture.m_flags, 0);

			release(mem);

			if (texture.m_depthFormat)
			{
				texture.m_depthOf = depthOf;
			}

			if (depth
			&& (texture.isTarget() || texture.m_depthFormat) )
			{
				texture.makeTarget(true, stencil, sampleable);
			}
		}

		/// No native handle is handed out: eacp's are its own business, and
		/// overrideInternal has nothing to swap in.
		uintptr_t getInternal(TextureHandle /*_handle*/) override
		{
			return 0;
		}

		void destroyTexture(TextureHandle _handle) override
		{
			m_textures[_handle.idx].destroy();
		}

		/// A framebuffer of textures. Up to maxFBAttachments colour attachments
		/// and one depth attachment, each at its own mip and layer (or run of
		/// layers, which is layered rendering): the pass renders into those
		/// images of the colour textures (Frame::beginMultiTargetPass over
		/// ColorTargets). The depth texture gets a carrier target of its own,
		/// of its kind and size with a dummy colour, whose plane the pass
		/// attaches at the depth attachment's image (RenderPassDescriptor::
		/// depthTarget), so a 2D depth texture goes beside an array, cube or
		/// volume colour, and depth sits at another level or layer than
		/// colour (G72, G74); a depth-only framebuffer renders into that
		/// carrier. A multisampled framebuffer, which depthTarget does not
		/// take (G88), has its depth on colour 0's own plane, colour 0 remade
		/// with one (G36). Either way the depth texture remembers which
		/// target carries its plane, for sampling. The images must be one
		/// size and one sample count, and the runs one length, as
		/// Frame::beginMultiTargetPass asks; one texture may appear several
		/// times at different images (G71).
		void createFrameBuffer(FrameBufferHandle _handle, uint8_t _num, const Attachment* _attachment) override
		{
			FrameBufferEacp& fb = m_frameBuffers[_handle.idx];
			fb = FrameBufferEacp();

			for (uint8_t ii = 0; ii < _num; ++ii)
			{
				const Attachment& at = _attachment[ii];

				if (!isValid(at.handle) )
				{
					continue;
				}

				FrameBufferEacp::Sub sub;
				sub.m_level  = at.mip;
				sub.m_slice  = at.layer;
				sub.m_layers = bx::max<uint16_t>(1, at.numLayers);

				if (1 < sub.m_layers
				&&  !GPU::Device::shared().supportsLayeredRendering() )
				{
					traceOnce(Unsupported::FrameBufferLayer, "a framebuffer attachment of several layers on a device without layered rendering renders into the first.");
					sub.m_layers = 1;
				}

				if (m_textures[at.handle.idx].m_depthFormat)
				{
					fb.m_depth    = at.handle.idx;
					fb.m_depthSub = sub;
				}
				else if (fb.m_num < g_caps.limits.maxFBAttachments)
				{
					fb.m_autoGen[fb.m_num] = 0 != (at.flags & BGFX_ATTACHMENT_AUTO_GEN_MIPS);
					fb.m_sub[fb.m_num]     = sub;
					fb.m_color[fb.m_num++] = at.handle.idx;
				}
			}

			for (uint8_t ii = 0; ii < fb.m_num; ++ii)
			{
				const TextureEacp& color = m_textures[fb.m_color[ii] ];

				if (!color.isTarget()
				||  !isRenderable(color.m_format) )
				{
					traceOnce(Unsupported::FrameBufferFormat, "G4: a framebuffer with a colour texture eacp cannot render into (R8, RG8, a refused format) is not drawn.");
					return;
				}

				const TextureEacp& first = m_textures[fb.m_color[0] ];
				const FrameBufferEacp::Sub& sub      = fb.m_sub[ii];
				const FrameBufferEacp::Sub& firstSub = fb.m_sub[0];

				bool duplicate = false;

				for (uint8_t jj = 0; jj < ii; ++jj)
				{
					duplicate |= fb.m_color[jj] == fb.m_color[ii] && fb.m_sub[jj] == sub;
				}

				if (duplicate
				||  !hasImage(color, sub)
				||  mipExtent(color.m_width,  sub.m_level) != mipExtent(first.m_width,  firstSub.m_level)
				||  mipExtent(color.m_height, sub.m_level) != mipExtent(first.m_height, firstSub.m_level)
				||  sub.m_layers != firstSub.m_layers
				||  color.m_texture->sampleCount() != first.m_texture->sampleCount() )
				{
					traceOnce(Unsupported::FrameBufferColor, "a framebuffer whose colour attachments differ in size, sample count or layers, name an image their texture lacks, or repeat one image, is not drawn (eacp refuses the pass).");
					return;
				}
			}

			if (0 < fb.m_num)
			{
				fb.m_target = fb.m_color[0];
			}
			else if (kInvalidHandle != fb.m_depth)
			{
				fb.m_target = fb.m_depth;
			}
			else
			{
				return;
			}

			if (kInvalidHandle != fb.m_depth)
			{
				TextureEacp& depth = m_textures[fb.m_depth];
				const bool sampleable = 0 == (depth.m_flags & BGFX_TEXTURE_RT_WRITE_ONLY);

				// The depth texture's own plane wherever depthTarget takes it:
				// single-sampled, its image the size of the pass's, and the
				// layer run inside it.
				bool own = 0 == fb.m_num;

				if (!own)
				{
					const TextureEacp& first = m_textures[fb.m_color[0] ];
					const FrameBufferEacp::Sub& firstSub = fb.m_sub[0];

					own = true
						&& 1 == first.m_texture->sampleCount()
						&& 1 == depth.m_desc.sampleCount
						&& 0 == fb.m_depthSub.m_level
						&& depth.m_width  == mipExtent(first.m_width,  firstSub.m_level)
						&& depth.m_height == mipExtent(first.m_height, firstSub.m_level)
						&& uint32_t(fb.m_depthSub.m_slice) + firstSub.m_layers <= depth.slices()
						;
				}

				if (own)
				{
					if (!depth.makeTarget(true, depth.m_stencilFormat, sampleable) )
					{
						return;
					}

					const FrameBufferEacp::Sub run = { fb.m_depthSub.m_level, fb.m_depthSub.m_slice, 0 == fb.m_num ? fb.m_depthSub.m_layers : fb.m_sub[0].m_layers };

					if (!hasImage(depth, run) )
					{
						traceOnce(Unsupported::FrameBufferColor, "a framebuffer whose colour attachments differ in size, sample count or layers, name an image their texture lacks, or repeat one image, is not drawn (eacp refuses the pass).");
						return;
					}

					depth.m_depthOf = fb.m_depth;
					fb.m_ownDepth   = 0 < fb.m_num;
				}
				else
				{
					// Multisampled, or a depth image that is not the pass's
					// size: colour 0's own plane, at colour 0's image.
					TextureEacp& target = m_textures[fb.m_target];

					if (kInvalidHandle != depth.m_depthOf
					&&  depth.m_depthOf != fb.m_target
					&&  m_textures[depth.m_depthOf].isTarget() )
					{
						traceOnce(Unsupported::FrameBufferDepth, "a multisampled depth texture shared by framebuffers with different colour textures is a separate plane in each; sampling it reads the latest.");
					}

					if (!(fb.m_depthSub == fb.m_sub[0]) )
					{
						traceOnce(Unsupported::FrameBufferDepth, "G88: a multisampled depth attachment, or one of another size than the pass, is colour 0's plane at colour 0's image.");
					}

					if (GPU::TextureType::TwoD == depth.m_kind
					&&  GPU::TextureType::TwoD != target.m_kind)
					{
						traceOnce(Unsupported::FrameBufferDepth, "a 2D depth texture of another size than the array, cube or volume colour beside it is that colour's own plane, which samples as an array.");
					}

					if (!target.makeTarget(true, depth.m_stencilFormat, sampleable) )
					{
						return;
					}

					depth.m_depthOf = fb.m_target;
				}
			}

			fb.m_valid = true;
		}

		/// Whether a render target has the level and the run of slices (a
		/// volume's z at that level) an attachment names; a run on a volume
		/// starts at z 0.
		static bool hasImage(const TextureEacp& _texture, const FrameBufferEacp::Sub& _sub)
		{
			if (!_texture.m_texture
			||  _sub.m_level >= _texture.m_texture->mipLevels() )
			{
				return false;
			}

			const uint32_t extent = GPU::TextureType::ThreeD == _texture.m_texture->kind()
				? mipExtent(uint32_t(_texture.m_texture->volumeDepth() ), _sub.m_level)
				: uint32_t(_texture.m_texture->slices() )
				;

			if (GPU::TextureType::ThreeD == _texture.m_texture->kind()
			&&  1 < _sub.m_layers
			&&  0 != _sub.m_slice)
			{
				return false;
			}

			return uint32_t(_sub.m_slice) + _sub.m_layers <= extent;
		}

		void createFrameBuffer(FrameBufferHandle _handle, const SwapChain& /*_desc*/) override
		{
			traceOnce(Unsupported::SwapChain, "G16: a framebuffer on a second window is not created; its views are skipped.");
			m_frameBuffers[_handle.idx] = FrameBufferEacp();
		}

		void destroyFrameBuffer(FrameBufferHandle _handle) override
		{
			m_frameBuffers[_handle.idx] = FrameBufferEacp();
		}

		/// The framebuffer's target texture (the one carrying the depth plane),
		/// or NULL when a view on it cannot be drawn.
		TextureEacp* frameBufferTarget(uint16_t _fbh)
		{
			const FrameBufferEacp& fb = m_frameBuffers[_fbh];

			if (!fb.m_valid
			||  !m_textures[fb.m_target].isTarget()
			|| (fb.m_ownDepth && !m_textures[fb.m_depth].isTarget() ) )
			{
				return NULL;
			}

			for (uint8_t ii = 0; ii < fb.m_num; ++ii)
			{
				if (!m_textures[fb.m_color[ii] ].isTarget() )
				{
					return NULL;
				}
			}

			return &m_textures[fb.m_target];
		}

		/// Texture::read, valid once the work that drew the texture has been
		/// committed: inside the frame (bgfx asks for screenshots and reads
		/// after submit, before the eacp frame ends) the frame is flushed first.
		void readBack(TextureEacp& _texture, void* _data)
		{
			flushFrame();
			_texture.read(_data, 0, 0);
		}

		/// Commits what this frame has recorded so far, so a read sees it.
		void flushFrame()
		{
			if (NULL != s_frame
			&&  m_needFlush)
			{
				endPass();
				s_frame->flush();
				m_needFlush = false;
			}
		}

		/// A screenshot of a texture framebuffer through Texture::read. The
		/// window's is G34: the drawable is not readable from inside the frame
		/// (eacp's Frame hands out neither it nor a read of it, and
		/// View::renderToImage would render the frame again, re-entering
		/// bgfx::renderFrame).
		void requestScreenShot(FrameBufferHandle _handle, const char* _filePath) override
		{
			if (!isValid(_handle) )
			{
				traceOnce(Unsupported::ScreenShotWindow, "G34: a screenshot of the window is not taken (no read of the drawable inside a frame).");
				return;
			}

			TextureEacp* target = frameBufferTarget(_handle.idx);

			if (NULL == target
			||  target->m_depthFormat)
			{
				return;
			}

			const uint32_t width  = target->m_width;
			const uint32_t height = target->m_height;
			const uint32_t pitch  = width*bimg::getBitsPerPixel(bimg::TextureFormat::Enum(target->m_textureFormat) )/8;
			const uint32_t size   = height*pitch;
			uint8_t* data = (uint8_t*)bx::alloc(g_allocator, size);

			readBack(*target, data);

			g_callback->screenShot(
				  _filePath
				, width
				, height
				, pitch
				, target->m_textureFormat
				, data
				, size
				, false
				);

			bx::free(g_allocator, data);
		}

		void invalidateOcclusionQuery(OcclusionQueryHandle /*_handle*/) override
		{
		}

		void setMarker(const char* /*_marker*/, uint16_t /*_len*/) override
		{
		}

		virtual void setName(Handle /*_handle*/, const char* /*_name*/, uint16_t /*_len*/) override
		{
		}

		// ---- uniforms -------------------------------------------------------------

		void setShaderUniform(uint8_t _flags, uint32_t _regIndex, const void* _val, uint32_t _numRegs)
		{
			uint8_t* scratch = 0 != (_flags & kUniformFragmentBit) ? m_fsScratch : m_vsScratch;

			if (_regIndex + _numRegs*16 <= sizeof(m_vsScratch) )
			{
				bx::memCopy(&scratch[_regIndex], _val, _numRegs*16);
			}
		}

		void setShaderUniform4f(uint8_t _flags, uint32_t _regIndex, const void* _val, uint32_t _numRegs)
		{
			setShaderUniform(_flags, _regIndex, _val, _numRegs);
		}

		void setShaderUniform4x4f(uint8_t _flags, uint32_t _regIndex, const void* _val, uint32_t _numRegs)
		{
			setShaderUniform(_flags, _regIndex, _val, _numRegs);
		}

		/// The program's user uniforms from bgfx's per-uniform values into the
		/// stage blocks, as renderer_vk.cpp's commit.
		void commit(UniformBuffer& _uniformBuffer)
		{
			_uniformBuffer.reset();

			for (;;)
			{
				uint32_t opcode = _uniformBuffer.read();

				if (UniformType::End == opcode)
				{
					break;
				}

				uint8_t type;
				uint16_t loc;
				uint16_t num;
				uint16_t copy;
				UniformBuffer::decodeOpcode(opcode, type, loc, num, copy);

				const char* data;
				if (copy)
				{
					data = _uniformBuffer.read(g_uniformTypeSize[type]*num);
				}
				else
				{
					UniformHandle handle;
					bx::memCopy(&handle, _uniformBuffer.read(sizeof(UniformHandle) ), sizeof(UniformHandle) );
					data = (const char*)m_uniforms[handle.idx];
				}

				if (NULL == data)
				{
					continue;
				}

				switch (type & ~kUniformFragmentBit)
				{
				case UniformType::Mat3:
					{
						const float* value = (const float*)data;
						for (uint32_t ii = 0, count = num/3; ii < count; ++ii, loc += 3*16, value += 9)
						{
							Matrix4 mtx;
							mtx.un.val[ 0] = value[0];
							mtx.un.val[ 1] = value[1];
							mtx.un.val[ 2] = value[2];
							mtx.un.val[ 3] = 0.0f;
							mtx.un.val[ 4] = value[3];
							mtx.un.val[ 5] = value[4];
							mtx.un.val[ 6] = value[5];
							mtx.un.val[ 7] = 0.0f;
							mtx.un.val[ 8] = value[6];
							mtx.un.val[ 9] = value[7];
							mtx.un.val[10] = value[8];
							mtx.un.val[11] = 0.0f;
							setShaderUniform(type, loc, &mtx.un.val[0], 3);
						}
					}
					break;

				case UniformType::Vec4:
				case UniformType::Mat4:
					setShaderUniform(type, loc, data, num);
					break;

				default:
					break;
				}
			}
		}

		void submitUniformCache(UniformCacheState& _ucs, uint16_t _view)
		{
			while (_ucs.hasItem(_view) )
			{
				const UniformCacheItem& uci = _ucs.advance();

				bx::memCopy(m_uniforms[uci.m_handle], &_ucs.m_frame->m_uniformCacheFrame.m_data[uci.m_offset], uci.m_size);
			}
		}

		static uint32_t blockBytes(const ShaderEacp* _shader)
		{
			if (NULL == _shader)
			{
				return 0;
			}

			if (_shader->m_size > 4096)
			{
				traceOnce(Unsupported::UniformBlockSize, "G9: a uniform block over 4 KB is clamped to 4 KB.");
			}

			return bx::min<uint32_t>(_shader->m_size, 4096);
		}

		void bindUniforms(GPU::RenderPass& _pass, const ProgramEacp& _program)
		{
			const uint32_t vsize = blockBytes(_program.m_vsh);
			const uint32_t fsize = blockBytes(_program.m_fsh);

			if (0 != vsize)
			{
				_pass.setVertexBytes(m_vsScratch, int(vsize), 0);
			}

			if (0 != fsize)
			{
				_pass.setFragmentBytes(m_fsScratch, int(fsize), 0);
			}
		}

		// ---- textures and samplers ---------------------------------------------------

		/// The palette entry as the nearest of eacp's three border colours, as
		/// renderer_mtl.cpp's getBorderColor (Metal has the same three).
		GPU::SamplerBorderColor borderColor(uint32_t _index)
		{
			static const float s_borderColor[][4] =
			{
				{ 0.0f, 0.0f, 0.0f, 0.0f },
				{ 0.0f, 0.0f, 0.0f, 1.0f },
				{ 1.0f, 1.0f, 1.0f, 1.0f },
			};

			if (NULL == m_colorPalette)
			{
				return GPU::SamplerBorderColor::OpaqueBlack;
			}

			const float* rgba = m_colorPalette[_index];

			uint32_t result = 1;
			float nearest = bx::kFloatLargest;

			for (uint32_t ii = 0; ii < BX_COUNTOF(s_borderColor); ++ii)
			{
				float distSq = 0.0f;
				for (uint32_t jj = 0; jj < 4; ++jj)
				{
					const float diff = rgba[jj] - s_borderColor[ii][jj];
					distSq += diff*diff;
				}

				if (distSq < nearest)
				{
					nearest = distSq;
					result  = ii;
				}
			}

			if (0.0f != nearest)
			{
				traceOnce(Unsupported::BorderColor, "a border colour other than transparent black, opaque black or opaque white is drawn as the nearest of them.");
			}

			return GPU::SamplerBorderColor(result);
		}

		/// bgfx's sampler flags as an eacp sampler, cached on the flags, the
		/// LOD clamp and the border colour the palette resolves to.
		const GPU::Sampler& getSampler(uint32_t _flags, uint8_t _lodMin, uint8_t _lodMax)
		{
			static const GPU::SamplerAddressMode s_addressMode[] =
			{
				GPU::SamplerAddressMode::Repeat,
				GPU::SamplerAddressMode::MirrorRepeat,
				GPU::SamplerAddressMode::ClampToEdge,
				GPU::SamplerAddressMode::ClampToBorder,
			};

			// [MIN|MAG] 0 linear, 1 point, 2 anisotropic (linear with anisotropy).
			static const GPU::SamplerFilter s_filter[] =
			{
				GPU::SamplerFilter::Linear,
				GPU::SamplerFilter::Nearest,
				GPU::SamplerFilter::Linear,
				GPU::SamplerFilter::Linear,
			};

			const uint32_t index = bx::min<uint32_t>(
				  BGFX_CONFIG_MAX_COLOR_PALETTE-1
				, (_flags & BGFX_SAMPLER_BORDER_COLOR_MASK) >> BGFX_SAMPLER_BORDER_COLOR_SHIFT
				);

			_flags &= BGFX_SAMPLER_BITS_MASK;

			const bool needBorderColor = false
				|| BGFX_SAMPLER_U_BORDER == (_flags & BGFX_SAMPLER_U_MASK)
				|| BGFX_SAMPLER_V_BORDER == (_flags & BGFX_SAMPLER_V_MASK)
				|| BGFX_SAMPLER_W_BORDER == (_flags & BGFX_SAMPLER_W_MASK)
				;

			const GPU::SamplerBorderColor border = needBorderColor
				? borderColor(index)
				: GPU::SamplerBorderColor::TransparentBlack
				;

			const bool anisotropic = true
				&& 0 != (m_reset & BGFX_RESET_MAXANISOTROPY)
				&& 0 != (_flags & (BGFX_SAMPLER_MIN_ANISOTROPIC|BGFX_SAMPLER_MAG_ANISOTROPIC) )
				;

			const uint64_t key = 0
				| uint64_t(_flags)
				| (uint64_t(border)      << 28)
				| (uint64_t(anisotropic) << 30)
				| (uint64_t(_lodMax)     << 32)
				| (uint64_t(_lodMin)     << 40)
				;

			auto it = m_samplers.find(key);

			if (it == m_samplers.end() )
			{
				GPU::SamplerDescriptor desc;
				desc.addressU      = s_addressMode[(_flags & BGFX_SAMPLER_U_MASK) >> BGFX_SAMPLER_U_SHIFT];
				desc.addressV      = s_addressMode[(_flags & BGFX_SAMPLER_V_MASK) >> BGFX_SAMPLER_V_SHIFT];
				desc.addressW      = s_addressMode[(_flags & BGFX_SAMPLER_W_MASK) >> BGFX_SAMPLER_W_SHIFT];
				desc.minFilter     = s_filter[(_flags & BGFX_SAMPLER_MIN_MASK) >> BGFX_SAMPLER_MIN_SHIFT];
				desc.magFilter     = s_filter[(_flags & BGFX_SAMPLER_MAG_MASK) >> BGFX_SAMPLER_MAG_SHIFT];
				desc.mipFilter     = 0 != (_flags & BGFX_SAMPLER_MIP_POINT) ? GPU::SamplerMipFilter::Nearest : GPU::SamplerMipFilter::Linear;
				desc.maxAnisotropy = anisotropic ? 16 : 1;
				desc.borderColor   = border;
				desc.lodMinClamp   = float(_lodMin) * 0.25f;
				desc.lodMaxClamp   = UINT8_MAX == _lodMax ? bx::kFloatLargest : float(_lodMax) * 0.25f;

				const uint32_t cmpFunc = (_flags & BGFX_SAMPLER_COMPARE_MASK) >> BGFX_SAMPLER_COMPARE_SHIFT;
				if (0 != cmpFunc)
				{
					desc.compare = s_cmpFunc[cmpFunc];
				}

				it = m_samplers.emplace(key, GPU::Device::shared().makeSampler(desc) ).first;
			}

			return it->second;
		}

		/// Every texture one shader stage declares, from the draw's binding
		/// at that stage, or the fallback when the draw has none there or its
		/// texture could not be made (or is the wrong dimension). A sampler
		/// entry without kUniformFragmentBit is the vertex shader's, bound
		/// through the vertex siblings of the fragment binds (G17).
		void bindTextures(GPU::RenderPass& _pass, const ShaderEacp& _shader, const RenderBind& _renderBind, bool _vertex)
		{
			for (uint8_t ii = 0; ii < _shader.m_numSamplers; ++ii)
			{
				const ShaderEacp::Sampler& slot = _shader.m_samplers[ii];

				if (slot.m_stage >= g_caps.limits.maxTextureSamplers)
				{
					traceOnce(Unsupported::SamplerSlot, "a texture past the device's texture slots (Device::maxTextureSlots) is not bound.");
					continue;
				}

				const GPU::TextureType kind = samplerKind(slot.m_dimension);
				const Binding& bind = _renderBind.m_bind[slot.m_stage];

				const TextureEacp* texture = NULL;
				uint32_t flags = 0;
				uint8_t lodMin = 0;
				uint8_t lodMax = UINT8_MAX;

				if (kInvalidHandle != bind.m_idx
				&&  Binding::Texture == bind.m_type)
				{
					texture = &m_textures[bind.m_idx];
					flags   = 0 == (BGFX_SAMPLER_INTERNAL_DEFAULT & bind.m_samplerFlags)
						? bind.m_samplerFlags
						: uint32_t(texture->m_flags)
						;
					lodMin = bind.m_lod.min;
					lodMax = bind.m_lod.max;

					if (texture->m_depthFormat)
					{
						// The depth plane, on whichever target carries it.
						const TextureEacp* carrier = depthCarrier(bind.m_idx);

						if (NULL == carrier)
						{
							traceOnce(Unsupported::DepthUnbound, "a depth texture no framebuffer has rendered (or sampled while its target is the open pass) reads a far-plane fallback.");
							carrier = &m_fallbackDepth;
						}

						const GPU::Sampler& sampler = getSampler(flags, lodMin, lodMax);
						const int slice = depthSlice(*carrier, kind, bind);

						if (0 <= slice)
						{
							// A 2D sampler on one layer or face of an array or
							// cube depth texture: that slice as a depth2d.
							if (_vertex)
							{
								_pass.setVertexDepthTextureSlice(*carrier->m_texture, slice, slot.m_stage, sampler);
							}
							else
							{
								_pass.setFragmentDepthTextureSlice(*carrier->m_texture, slice, slot.m_stage, sampler);
							}
						}
						else if (_vertex)
						{
							_pass.setVertexDepthTexture(*carrier->m_texture, slot.m_stage, sampler);
						}
						else
						{
							_pass.setFragmentDepthTexture(*carrier->m_texture, slot.m_stage, sampler);
						}

						continue;
					}
					else if (!texture->m_texture
					     ||  isPassAttachment(bind.m_idx) )
					{
						texture = NULL;
					}
					else if (texture->m_kind != kind)
					{
						// Metal samples nothing through a texture of another
						// type than the shader declared, and reports it only
						// under the validation layer.
						traceOnce(Unsupported::SamplerKind, "a texture of another kind than the shader samples (2D, cube, array, volume) reads the white fallback.");
						texture = NULL;
					}
				}

				if (NULL == texture)
				{
					texture = &m_fallback[uint32_t(kind)];
					flags   = uint32_t(texture->m_flags);
				}

				if (!texture->m_texture)
				{
					continue;
				}

				const GPU::Sampler& sampler = getSampler(flags, lodMin, lodMax);

				if (_vertex)
				{
					_pass.setVertexTexture(*texture->m_texture, slot.m_stage, sampler);
				}
				else
				{
					_pass.setFragmentTexture(*texture->m_texture, slot.m_stage, sampler);
				}
			}
		}

		/// The slice of an array, cube or cube-array depth carrier a shader
		/// declaring a 2D sampler reads (the binding's first layer, a face of
		/// a cube), or -1 when the shader's kind is the carrier's own and the
		/// whole plane binds.
		static int depthSlice(const TextureEacp& _carrier, GPU::TextureType _kind, const Binding& _bind)
		{
			if (GPU::TextureType::TwoD != _kind
			||  GPU::TextureType::TwoD == _carrier.m_texture->kind() )
			{
				return -1;
			}

			const int slices = bx::max(1, _carrier.m_texture->slices() );
			return bx::min(int(_bind.m_firstLayer), slices - 1);
		}

		/// The target whose depth plane a depth texture is, when it can be
		/// sampled: made sampleable, and not the target of the open pass.
		const TextureEacp* depthCarrier(uint16_t _depth) const
		{
			const uint16_t idx = m_textures[_depth].m_depthOf;

			if (kInvalidHandle == idx
			||  idx == m_passTexture
			||  idx == m_passDepth)
			{
				return NULL;
			}

			const TextureEacp& carrier = m_textures[idx];

			if (!carrier.isTarget()
			||  !carrier.m_texture->hasSampleableDepth() )
			{
				return NULL;
			}

			return &carrier;
		}

		// ---- compute -------------------------------------------------------------

		/// A storage binding's buffer from its byte offset: the device buffer of a
		/// compute buffer, or this frame's copy of any other.
		GPU::BufferRange storageRange(const Binding& _bind)
		{
			BufferEacp* buffer = Binding::IndexBuffer == _bind.m_type
				? m_indexBuffers[_bind.m_idx]
				: m_vertexBuffers[_bind.m_idx]
				;

			if (NULL == buffer)
			{
				return {};
			}

			GPU::BufferRange range = buffer->range(buffer->m_vertex ? m_vertexStream : m_indexStream);
			const uint32_t offset = bx::min(_bind.m_offset, buffer->m_size);
			range.offset += offset;
			range.bytes   = bx::max<int64_t>(range.bytes - offset, 0);
			return range;
		}

		static bool hasImageBinding(const RenderBind& _renderBind)
		{
			for (uint32_t stage = 0; stage < BGFX_CONFIG_MAX_TEXTURE_SAMPLERS; ++stage)
			{
				const Binding& bind = _renderBind.m_bind[stage];

				if (kInvalidHandle != bind.m_idx
				&&  Binding::Image == bind.m_type)
				{
					return true;
				}
			}

			return false;
		}

		/// A render stage's storage buffers: eacp-metal declares stage n at
		/// buffer(24 + n) in both stages, RenderPass::bufferBase, which is where
		/// setVertexStorageBuffer and setFragmentStorageBuffer bind slot n.
		/// Metal's argument table ends at buffer(30), so a render stage has
		/// seven storage slots however many a kernel has.
		void bindRenderStorage(GPU::RenderPass& _pass, const RenderBind& _renderBind)
		{
			const uint32_t numStages = bx::min<uint32_t>(g_caps.limits.maxComputeBindings, 31 - GPU::RenderPass::bufferBase);

			for (uint8_t stage = 0; stage < numStages; ++stage)
			{
				const Binding& bind = _renderBind.m_bind[stage];

				if (kInvalidHandle == bind.m_idx
				|| (Binding::IndexBuffer  != bind.m_type
				&&  Binding::VertexBuffer != bind.m_type) )
				{
					continue;
				}

				const GPU::BufferRange range = storageRange(bind);

				if (range.isValid() )
				{
					_pass.setVertexStorageBuffer(range, stage);
					_pass.setFragmentStorageBuffer(range, stage);
				}
			}
		}

		/// One RenderCompute item: into the view's open compute pass (a render
		/// pass ends first), the kernel, its uniform block at buffer(16)
		/// (ComputePass::setBytes slot 0, where eacp-metal puts _mtl_u), and
		/// bgfx's bind stages as eacp-metal numbers them: a buffer at
		/// buffer(stage), an image or a texture at texture(stage), a texture's
		/// sampler at sampler(stage). bgfx dispatches threadgroups and eacp
		/// threads, so the grid is the group count times the kernel's group.
		void submitCompute(GPU::Frame& _frame, ViewState& _viewState, uint16_t _view, const View& _renderView, const char* _viewName, const RenderCompute& _compute, ProgramHandle _program, Frame* _render)
		{
			ProgramEacp* program = isValid(_program) ? m_program[_program.idx] : NULL;

			if (NULL == program
			||  !program->isCompute() )
			{
				return;
			}

			if (!m_computePass)
			{
				m_pass.reset();
				m_passTexture  = kInvalidHandle;
				m_passDepth    = kInvalidHandle;
				m_passNumColor = 0;
				m_computePass.emplace(ComputePassMaker{ _frame, _viewName });
				m_needFlush = true;
			}

			GPU::ComputePass& pass = *m_computePass;
			pass.setPipeline(*program->m_compute);

			rendererUpdateUniforms(this, _render->m_uniformBuffer[_compute.m_uniformIdx], _compute.m_uniformBegin, _compute.m_uniformEnd);

			if (NULL != program->m_vsh->m_constantBuffer)
			{
				commit(*program->m_vsh->m_constantBuffer);
			}

			_viewState.setPredefined<4>(this, _view, _renderView, *program, _render, _compute);

			const uint32_t size = blockBytes(program->m_vsh);

			if (0 != size)
			{
				pass.setBytes(m_vsScratch, int64_t(size), 0);
			}

			const RenderBind& renderBind = _render->m_renderBind[_compute.m_bindIdx];

			// A write image the kernel cannot be given (G67).
			bool unbound = false;

			for (uint8_t stage = 0; stage < g_caps.limits.maxComputeBindings; ++stage)
			{
				const Binding& bind = renderBind.m_bind[stage];

				if (kInvalidHandle == bind.m_idx)
				{
					continue;
				}

				switch (bind.m_type)
				{
				case Binding::IndexBuffer:
				case Binding::VertexBuffer:
					{
						const GPU::BufferRange range = storageRange(bind);

						if (!range.isValid() )
						{
							traceOnce(Unsupported::ComputeBind, "a compute binding of a buffer or texture that does not exist is left unbound.");
						}
						else if (Access::Read == bind.m_access)
						{
							pass.setInputBuffer(range, stage);
						}
						else
						{
							pass.setOutputBuffer(range, stage);
						}
					}
					break;

				case Binding::Image:
					{
						const TextureEacp& texture = m_textures[bind.m_idx];

						// setImage's mip is one level of the texture, every
						// slice of it (setOutputTextureLevel,
						// setInputTextureLevel); level 0 is the whole texture's
						// own bind. A cube or cube array is the 2D array of
						// its slices at that level (the AsArray binds), as
						// bgfx's IMAGE2D_ARRAY declares it. An image the kernel
						// would write and eacp cannot bind for writing (a
						// format the device does not compute-write, a level
						// the texture lacks) or read-write through one bind
						// (Device::supportsComputeReadWrite) leaves the slot to
						// whatever texture is there, which Metal's validation
						// stops on: the dispatch is skipped instead (G67).
						const uint8_t mip   = bind.m_firstMip;
						const bool    write = Access::Read != bind.m_access;
						const bool    cube  = GPU::TextureType::Cube == texture.m_kind || GPU::TextureType::CubeArray == texture.m_kind;

						if (!texture.m_texture
						||  mip >= texture.m_texture->mipLevels() )
						{
							traceOnce(Unsupported::ComputeBind, "a compute binding of a buffer, texture or mip level that does not exist is left unbound.");
							unbound |= write;
						}
						else if (Access::ReadWrite == bind.m_access
						     &&  !GPU::Device::shared().supportsComputeReadWrite(texture.m_format) )
						{
							traceOnce(Unsupported::ComputeReadWrite, "a kernel reading and writing an image in a format the device cannot read-write through one bind (Device::supportsComputeReadWrite) is not dispatched.");
							unbound = true;
						}
						else if (texture.m_texture->isComputeWritable() )
						{
							if (cube)
							{
								pass.setOutputTextureAsArray(*texture.m_texture, stage, int(mip) );
							}
							else if (0 == mip)
							{
								pass.setOutputTexture(*texture.m_texture, stage);
							}
							else
							{
								pass.setOutputTextureLevel(*texture.m_texture, stage, int(mip) );
							}
						}
						else if (!write)
						{
							// access::read needs only ShaderRead, which every
							// eacp texture has; the sampler bound beside it is
							// unused.
							if (cube)
							{
								pass.setInputTextureAsArray(*texture.m_texture, stage, int(mip) );
							}
							else if (0 == mip)
							{
								pass.setInputTexture(*texture.m_texture, stage, getSampler(BGFX_SAMPLER_POINT|BGFX_SAMPLER_UVW_CLAMP, 0, UINT8_MAX) );
							}
							else
							{
								pass.setInputTextureLevel(*texture.m_texture, stage, int(mip) );
							}
						}
						else
						{
							traceOnce(Unsupported::ComputeImageFormat, "G14: a kernel writing an image the device cannot compute-write (BGRA8, a block format) is not dispatched.");
							unbound = true;
						}
					}
					break;

				case Binding::Texture:
					{
						const TextureEacp* texture = &m_textures[bind.m_idx];
						uint32_t flags = 0 == (BGFX_SAMPLER_INTERNAL_DEFAULT & bind.m_samplerFlags)
							? bind.m_samplerFlags
							: uint32_t(texture->m_flags)
							;

						if (texture->m_depthFormat)
						{
							// The depth plane, on whichever target carries
							// it, as the render stages bind it
							// (setInputDepthTexture beside
							// setFragmentDepthTexture).
							const TextureEacp* carrier = depthCarrier(bind.m_idx);

							if (NULL == carrier)
							{
								traceOnce(Unsupported::DepthUnbound, "a depth texture no framebuffer has rendered (or sampled while its target is the open pass) reads a far-plane fallback.");
								carrier = &m_fallbackDepth;
							}

							int slice = -1;

							for (uint8_t ss = 0; ss < program->m_vsh->m_numSamplers; ++ss)
							{
								if (stage == program->m_vsh->m_samplers[ss].m_stage)
								{
									slice = depthSlice(*carrier, samplerKind(program->m_vsh->m_samplers[ss].m_dimension), bind);
								}
							}


							if (0 <= slice)
							{
								pass.setInputDepthTextureSlice(*carrier->m_texture, slice, stage, getSampler(flags, bind.m_lod.min, bind.m_lod.max) );
							}
							else
							{
								pass.setInputDepthTexture(*carrier->m_texture, stage, getSampler(flags, bind.m_lod.min, bind.m_lod.max) );
							}
							break;
						}
						else if (!texture->m_texture)
						{
							texture = NULL;
						}

						if (NULL == texture)
						{
							texture = &m_fallback[0];
							flags   = uint32_t(texture->m_flags);
						}

						pass.setInputTexture(*texture->m_texture, stage, getSampler(flags, bind.m_lod.min, bind.m_lod.max) );
					}
					break;

				default:
					break;
				}
			}

			if (unbound)
			{
				return;
			}

			if (isValid(_compute.m_indirectBuffer) )
			{
				// bgfx's indirect dispatch arguments are three uint32 group
				// counts every BGFX_CONFIG_DRAW_INDIRECT_STRIDE bytes, which is
				// eacp's DispatchArguments at an offset.
				BufferEacp* args = m_vertexBuffers[_compute.m_indirectBuffer.idx];

				if (NULL == args
				||  !args->m_static)
				{
					return;
				}

				const uint32_t numRecords  = args->m_size/BGFX_CONFIG_DRAW_INDIRECT_STRIDE;
				const uint32_t numIndirect = _compute.m_startIndirect < numRecords
					? bx::min(numRecords - _compute.m_startIndirect, _compute.m_numIndirect)
					: 0
					;

				uint32_t offset = _compute.m_startIndirect*BGFX_CONFIG_DRAW_INDIRECT_STRIDE;

				for (uint32_t ii = 0; ii < numIndirect; ++ii, offset += BGFX_CONFIG_DRAW_INDIRECT_STRIDE)
				{
					pass.dispatchIndirect(*args->m_static, int64_t(offset) );
				}

				return;
			}

			const uint16_t* group = program->m_vsh->m_numThreads;

			if (0 != _compute.m_numX
			&&  0 != _compute.m_numY
			&&  0 != _compute.m_numZ)
			{
				pass.dispatch(
					  int(_compute.m_numX*group[0])
					, int(_compute.m_numY*group[1])
					, int(_compute.m_numZ*group[2])
					);
			}
		}

		// ---- pipelines -----------------------------------------------------------

		static constexpr uint64_t kPipelineStateMask = 0
			| BGFX_STATE_WRITE_RGB
			| BGFX_STATE_WRITE_A
			| BGFX_STATE_WRITE_Z
			| BGFX_STATE_DEPTH_TEST_MASK
			| BGFX_STATE_BLEND_MASK
			| BGFX_STATE_BLEND_EQUATION_MASK
			| BGFX_STATE_BLEND_INDEPENDENT
			| BGFX_STATE_CULL_MASK
			| BGFX_STATE_FRONT_CCW
			| BGFX_STATE_PT_MASK
			;

		bool windowHasDepth() const
		{
			return NULL == m_view || m_view->hasDepth();
		}

		bool windowHasStencil() const
		{
			return NULL != m_view && m_view->hasStencil();
		}

		int windowSampleCount() const
		{
			return NULL == m_view ? 1 : m_view->sampleCount();
		}

		/// What the open pass renders into, which a pipeline has to agree with:
		/// colour format, sample count, and whether there is a depth and a
		/// stencil plane.
		struct Target
		{
			GPU::PixelFormat m_colorFormat[BGFX_CONFIG_MAX_FRAME_BUFFER_ATTACHMENTS] = { GPU::PixelFormat::BGRA8Unorm };
			uint8_t m_numColor = 1;
			int  m_sampleCount = 1;
			bool m_depth       = true;
			bool m_stencil     = false;
		};

		Target windowTarget() const
		{
			Target target;
			target.m_colorFormat[0] = GPU::PixelFormat::BGRA8Unorm;
			target.m_sampleCount    = windowSampleCount();
			target.m_depth          = windowHasDepth();
			target.m_stencil        = windowHasStencil();
			return target;
		}

		/// One texture, or a framebuffer's colour attachments with the depth
		/// and stencil of the first (G36).
		Target textureTarget(const TextureEacp& _texture, const FrameBufferEacp* _fb) const
		{
			Target target;
			target.m_colorFormat[0] = GPU::pixelFormatFor(_texture.m_format);
			target.m_sampleCount    = _texture.m_texture->sampleCount();
			target.m_depth          = _texture.m_texture->hasDepth();
			target.m_stencil        = _texture.m_texture->hasStencil();

			if (NULL != _fb
			&&  1 < _fb->m_num)
			{
				target.m_numColor = _fb->m_num;

				for (uint8_t ii = 0; ii < _fb->m_num; ++ii)
				{
					target.m_colorFormat[ii] = GPU::pixelFormatFor(m_textures[_fb->m_color[ii] ].m_format);
				}
			}

			return target;
		}

		static void translateStencil(GPU::StencilFace& _face, uint32_t _stencil)
		{
			_face.compare     = s_cmpFunc[(_stencil&BGFX_STENCIL_TEST_MASK)>>BGFX_STENCIL_TEST_SHIFT];
			_face.stencilFail = s_stencilOp[(_stencil&BGFX_STENCIL_OP_FAIL_S_MASK)>>BGFX_STENCIL_OP_FAIL_S_SHIFT];
			_face.depthFail   = s_stencilOp[(_stencil&BGFX_STENCIL_OP_FAIL_Z_MASK)>>BGFX_STENCIL_OP_FAIL_Z_SHIFT];
			_face.pass        = s_stencilOp[(_stencil&BGFX_STENCIL_OP_PASS_Z_MASK)>>BGFX_STENCIL_OP_PASS_Z_SHIFT];
		}

		/// The vertex layout a program's [[attribute(n)]] list asks for, out of
		/// the draw's streams (stream k at Metal buffer k) and its instance data
		/// (at buffer numStreams, 16 bytes per i_dataN).
		GPU::VertexLayout translateLayout(const ProgramEacp& _program, uint8_t _numStreams, const VertexLayout** _layouts, uint8_t _numInstanceData)
		{
			GPU::VertexLayout result;

			for (uint8_t stream = 0; stream < _numStreams; ++stream)
			{
				result.buffer(stream, _layouts[stream]->getStride(), GPU::StepRate::PerVertex);
			}

			if (0 < _numInstanceData)
			{
				result.buffer(_numStreams, _numInstanceData*16, GPU::StepRate::PerInstance);
			}

			if (result.buffers.empty() )
			{
				result.buffer(0, 4, GPU::StepRate::PerVertex);
			}

			for (uint32_t location = 0; location < _program.m_numAttributes; ++location)
			{
				const AttributeSource& source = _program.m_attributes[location];

				GPU::VertexFormat format = GPU::VertexFormat::UByte4Norm;
				int offset      = 0;
				int bufferIndex = 0;
				bool found      = false;

				if (AttributeSource::Vertex == source.m_kind)
				{
					const Attrib::Enum attr = Attrib::Enum(source.m_index);

					for (uint8_t stream = 0; stream < _numStreams && !found; ++stream)
					{
						const VertexLayout& layout = *_layouts[stream];

						if (layout.has(attr) )
						{
							uint8_t num;
							AttribType::Enum type;
							bool normalized;
							bool asInt;
							layout.decode(attr, num, type, normalized, asInt);

							format      = toVertexFormat(num, type, normalized, asInt);
							offset      = layout.m_offset[attr];
							bufferIndex = stream;
							found       = true;
						}
					}
				}
				else if (AttributeSource::Instance == source.m_kind
					 &&  source.m_index < _numInstanceData)
				{
					format      = GPU::VertexFormat::Float4;
					offset      = source.m_index*16;
					bufferIndex = _numStreams;
					found       = true;
				}

				if (!found)
				{
					// What the stream does not carry reads the first bytes of
					// stream 0 rather than failing the pipeline, as Metal's
					// fallback attribute does.
					BX_TRACE("eacp: attribute(%d) is not in the draw's streams.", location);
				}

				result.attribute(format, offset, bufferIndex);
			}

			return result;
		}

		/// The blend of one colour attachment from bgfx's packing: the state
		/// word's for attachment 0 (and every attachment without
		/// BGFX_STATE_BLEND_INDEPENDENT), 11 bits of _rgba per attachment after
		/// it, as renderer_mtl.cpp decodes them.
		static std::optional<GPU::BlendState> translateBlend(uint64_t _state, uint32_t _rgba, uint32_t _attachment)
		{
			uint32_t srcRGB, dstRGB, srcA, dstA, equRGB, equA;

			if (0 == _attachment
			||  0 == (_state & BGFX_STATE_BLEND_INDEPENDENT) )
			{
				if (0 == (_state & BGFX_STATE_BLEND_MASK) )
				{
					return std::nullopt;
				}

				const uint32_t blend    = uint32_t( (_state&BGFX_STATE_BLEND_MASK         )>>BGFX_STATE_BLEND_SHIFT);
				const uint32_t equation = uint32_t( (_state&BGFX_STATE_BLEND_EQUATION_MASK)>>BGFX_STATE_BLEND_EQUATION_SHIFT);

				srcRGB = (blend    )&0xf;
				dstRGB = (blend>> 4)&0xf;
				srcA   = (blend>> 8)&0xf;
				dstA   = (blend>>12)&0xf;
				equRGB = (equation   )&0x7;
				equA   = (equation>>3)&0x7;
			}
			else
			{
				const uint32_t shift = 11*(_attachment-1);
				const uint32_t rgba  = shift < 32 ? (_rgba>>shift)&0x7ff : 0;

				if (0 == rgba)
				{
					return std::nullopt;
				}

				srcRGB = srcA = (rgba   )&0xf;
				dstRGB = dstA = (rgba>>4)&0xf;
				equRGB = equA = (rgba>>8)&0x7;
			}

			for (uint32_t factor : { srcRGB, dstRGB, srcA, dstA })
			{
				if (factor >= 12)
				{
					traceOnce(Unsupported::BlendFactor, "G26: the constant blend factor (BGFX_STATE_BLEND_FACTOR) is drawn as One/Zero.");
				}
			}

			GPU::BlendState state;
			state.enabled          = true;
			state.sourceColor      = s_blendFactor[srcRGB][0];
			state.destinationColor = s_blendFactor[dstRGB][0];
			state.colorOperation   = s_blendEquation[bx::min<uint32_t>(equRGB, 4)];
			state.sourceAlpha      = s_blendFactor[srcA][1];
			state.destinationAlpha = s_blendFactor[dstA][1];
			state.alphaOperation   = s_blendEquation[bx::min<uint32_t>(equA, 4)];
			return state;
		}

		/// _rgba is the draw's per-attachment blend, read under
		/// BGFX_STATE_BLEND_INDEPENDENT only (otherwise it is the blend factor
		/// colour, G26); _mrtMask the attachments written, the others' write
		/// mask none (the clear quad's skipped attachments).
		const PipelineEacp* getPipeline(uint64_t _state, uint64_t _stencil, ProgramHandle _program, uint8_t _numStreams, const VertexLayout** _layouts, uint8_t _numInstanceData, uint32_t _rgba = 0, uint8_t _mrtMask = UINT8_MAX)
		{
			_state &= kPipelineStateMask;

			_rgba = 0 != (_state & BGFX_STATE_BLEND_INDEPENDENT) && 1 < m_target.m_numColor
				? _rgba
				: 0
				;

			// Bits past the target's attachments mean nothing; set, so that
			// one pipeline answers every mask that writes the same ones.
			_mrtMask = uint8_t(_mrtMask | ~( (1u<<m_target.m_numColor) - 1) );

			const bool hasStencil = m_target.m_stencil;

			_stencil = hasStencil && stencilEnabled(_stencil)
				? _stencil & kStencilNoRefMask
				: 0
				;

			// The draw before asked for the same pipeline into the same target
			// (m_lastPipeline is dropped whenever a pass begins): no hash, as
			// renderer_mtl.cpp skips its pipeline lookup while the state word
			// and program are unchanged.
			if (NULL != m_lastPipeline
			&&  _state           == m_lastQuery.m_state
			&&  _stencil         == m_lastQuery.m_stencil
			&&  _program.idx     == m_lastQuery.m_program
			&&  _numStreams      == m_lastQuery.m_numStreams
			&&  _numInstanceData == m_lastQuery.m_numInstanceData
			&&  _rgba            == m_lastQuery.m_rgba
			&&  _mrtMask         == m_lastQuery.m_mrtMask)
			{
				bool same = true;

				for (uint8_t ii = 0; ii < _numStreams && same; ++ii)
				{
					same = _layouts[ii]->m_hash == m_lastQuery.m_layoutHash[ii];
				}

				if (same)
				{
					return m_lastPipeline;
				}
			}

			m_lastQuery.m_state           = _state;
			m_lastQuery.m_stencil         = _stencil;
			m_lastQuery.m_program         = _program.idx;
			m_lastQuery.m_numStreams      = _numStreams;
			m_lastQuery.m_numInstanceData = _numInstanceData;
			m_lastQuery.m_rgba            = _rgba;
			m_lastQuery.m_mrtMask         = _mrtMask;

			for (uint8_t ii = 0; ii < _numStreams; ++ii)
			{
				m_lastQuery.m_layoutHash[ii] = _layouts[ii]->m_hash;
			}

			const ProgramEacp& program = *m_program[_program.idx];

			bx::HashMurmur3 murmur;
			murmur.begin();
			murmur.add(_state);
			murmur.add(_stencil);
			murmur.add(_program.idx);
			murmur.add(program.m_vsh->m_hash);
			murmur.add(NULL != program.m_fsh ? program.m_fsh->m_hash : 0);
			murmur.add(_numStreams);
			murmur.add(_numInstanceData);
			murmur.add(_rgba);
			murmur.add(_mrtMask);
			murmur.add(m_target.m_numColor);
			murmur.add(m_target.m_sampleCount);
			murmur.add(m_target.m_depth);
			murmur.add(m_target.m_stencil);

			for (uint8_t ii = 0; ii < m_target.m_numColor; ++ii)
			{
				murmur.add(uint32_t(m_target.m_colorFormat[ii]) );
			}

			for (uint8_t ii = 0; ii < _numStreams; ++ii)
			{
				murmur.add(_layouts[ii]->m_hash);
			}

			const uint32_t key = murmur.end() | 1;

			if (key == m_lastPipelineKey)
			{
				return m_lastPipeline;
			}

			auto it = m_pipelines.find(key);

			if (it == m_pipelines.end() )
			{
				PipelineEacp* entry = BX_NEW(g_allocator, PipelineEacp);
				entry->m_program = _program.idx;

				GPU::RenderPipelineDescriptor desc;
				desc.library      = &*program.m_library;
				desc.vertexLayout = translateLayout(program, _numStreams, _layouts, _numInstanceData);
				desc.colorFormat  = m_target.m_colorFormat[0];
				desc.sampleCount  = m_target.m_sampleCount;

				desc.topology = s_primInfo[(_state&BGFX_STATE_PT_MASK)>>BGFX_STATE_PT_SHIFT].m_type;

				GPU::ColorWriteMask writeMask;
				writeMask.red   = 0 != (_state & BGFX_STATE_WRITE_R);
				writeMask.green = 0 != (_state & BGFX_STATE_WRITE_G);
				writeMask.blue  = 0 != (_state & BGFX_STATE_WRITE_B);
				writeMask.alpha = 0 != (_state & BGFX_STATE_WRITE_A);

				desc.colorWriteMask = 0 != (_mrtMask & 1) ? writeMask : GPU::ColorWriteMask::none();
				// An integer attachment (R32U) takes no blending.
				desc.blend          = GPU::PixelFormat::R32Uint != desc.colorFormat ? translateBlend(_state, _rgba, 0) : std::nullopt;

				// Attachments 1 and up: their own format, bgfx's one write mask,
				// and the state word's blend or, under
				// BGFX_STATE_BLEND_INDEPENDENT, their own from _rgba.
				for (uint8_t ii = 1; ii < m_target.m_numColor; ++ii)
				{
					GPU::ColorTargetState target;
					target.format    = m_target.m_colorFormat[ii];
					target.blend     = GPU::PixelFormat::R32Uint != target.format ? translateBlend(_state, _rgba, ii) : std::nullopt;
					target.writeMask = 0 != (_mrtMask & (1<<ii) ) ? writeMask : GPU::ColorWriteMask::none();
					desc.extraColorTargets.push_back(target);
				}

				// A pipeline must agree with the pass about whether there is a
				// depth plane, so every pipeline into a target with one has it
				// and "no depth test" is Always.
				desc.depth        = m_target.m_depth;
				desc.depthCompare = s_cmpFunc[(_state&BGFX_STATE_DEPTH_TEST_MASK)>>BGFX_STATE_DEPTH_TEST_SHIFT];
				desc.depthWrite   = 0 != (_state & BGFX_STATE_WRITE_Z);

				const uint32_t cull = uint32_t( (_state&BGFX_STATE_CULL_MASK)>>BGFX_STATE_CULL_SHIFT);
				desc.cullMode  = 1 == cull ? GPU::CullMode::Front
					: 2 == cull ? GPU::CullMode::Back
					: GPU::CullMode::None
					;
				desc.frontFace = 0 != (_state & BGFX_STATE_FRONT_CCW)
					? GPU::Winding::CounterClockwise
					: GPU::Winding::Clockwise
					;

				desc.stencil = hasStencil;

				if (hasStencil
				&&  0 != _stencil)
				{
					const uint32_t fstencil = unpackStencil(0, _stencil);
					const uint32_t bstencil = stencilFrontAndBack(_stencil) ? unpackStencil(1, _stencil) : fstencil;

					translateStencil(desc.stencilFront, fstencil);
					translateStencil(desc.stencilBack,  bstencil);

					// G18: one read and one write mask for both faces.
					desc.stencilReadMask  = uint8_t( (fstencil&BGFX_STENCIL_FUNC_RMASK_MASK)>>BGFX_STENCIL_FUNC_RMASK_SHIFT);
					desc.stencilWriteMask = unpackStencilWriteMask(_stencil);
				}

				entry->m_pipeline.emplace(GPU::Device::shared().makeRenderPipeline(desc) );

				if (!entry->m_pipeline->isValid() )
				{
					BX_TRACE("eacp: pipeline for program %d failed (state 0x%016" PRIx64 ").", _program.idx, _state);
				}

				it = m_pipelines.emplace(key, entry).first;
			}

			m_lastPipelineKey = key;
			m_lastPipeline    = it->second;

			return it->second;
		}

		// ---- the frame -------------------------------------------------------------

		/// A view's clear inside an open pass, where the pass's own clear would
		/// clear the whole target: bgfx's clear quad, scissored to the view.
		void clearQuad(GPU::RenderPass& _pass, const ClearQuad& _clearQuad, const Clear& _clear, const float _palette[][4])
		{
			// bgfx's clear program for N attachments writes
			// bgfx_clear_color[0..N-1], one colour per attachment.
			const uint32_t numMrt = m_target.m_numColor;
			const ProgramHandle programHandle = _clearQuad.m_program[numMrt-1];

			if (!isValid(programHandle)
			||  NULL == m_program[programHandle.idx]
			||  !m_program[programHandle.idx]->m_library
			||  NULL == m_vertexBuffers[_clearQuad.m_vb.idx])
			{
				return;
			}

			const uint64_t state = 0
				| (_clear.m_flags & BGFX_CLEAR_COLOR ? BGFX_STATE_WRITE_RGB|BGFX_STATE_WRITE_A         : 0)
				| (_clear.m_flags & BGFX_CLEAR_DEPTH ? BGFX_STATE_DEPTH_TEST_ALWAYS|BGFX_STATE_WRITE_Z : 0)
				| BGFX_STATE_PT_TRISTRIP
				;

			const uint64_t stencil = _clear.m_flags & BGFX_CLEAR_STENCIL
				? packStencil(0
					| BGFX_STENCIL_TEST_ALWAYS
					| BGFX_STENCIL_FUNC_REF(_clear.m_stencil)
					| BGFX_STENCIL_FUNC_RMASK(0xff)
					| BGFX_STENCIL_OP_FAIL_S_REPLACE
					| BGFX_STENCIL_OP_FAIL_Z_REPLACE
					| BGFX_STENCIL_OP_PASS_Z_REPLACE
					, BGFX_STENCIL_NONE)
				: 0
				;

			const VertexLayout* layout = &m_vertexLayouts[_clearQuad.m_layout.idx];
			const PipelineEacp* pipeline = getPipeline(state, stencil, programHandle, 1, &layout, 0, 0, uint8_t(~_clear.getColorSkipMask(numMrt) ) );

			if (!pipeline->m_pipeline->isValid() )
			{
				return;
			}

			const ProgramEacp& program = *m_program[programHandle.idx];

			const float depth[4] = { _clear.m_depth };
			float rgba[BGFX_CONFIG_MAX_FRAME_BUFFER_ATTACHMENTS][4];

			for (uint32_t ii = 0; ii < numMrt; ++ii)
			{
				const ::eacp::Graphics::Color color = clearColor(_clear, _palette, ii);
				rgba[ii][0] = color.r;
				rgba[ii][1] = color.g;
				rgba[ii][2] = color.b;
				rgba[ii][3] = color.a;
			}

			bx::memCopy(m_vsScratch, depth, sizeof(depth) );
			bx::memCopy(m_fsScratch, rgba, numMrt*sizeof(rgba[0]) );

			_pass.setPipeline(*pipeline->m_pipeline);
			_pass.setStencilReference(_clear.m_stencil);
			bindUniforms(_pass, program);
			_pass.setVertexBuffer(m_vertexBuffers[_clearQuad.m_vb.idx]->range(m_vertexStream), 0);
			_pass.draw(4);
		}

		void submit(Frame* _render, const ClearQuad& _clearQuad, const MipGen& /*_mipGen*/, TextVideoMemBlitter& _textVideoMemBlitter) override
		{
			const int64_t timerFreq = bx::getHPFrequency();
			const int64_t timeBegin = bx::getHPCounter();

			uint32_t statsKeyType[2] = {};
			uint32_t statsNumPrimsRendered[BX_COUNTOF(s_primInfo)] = {};

			m_colorPalette = _render->m_colorPalette;
			m_reset        = _render->m_reset;

			if (0 < _render->m_iboffset)
			{
				TransientIndexBuffer* ib = _render->m_transientIb;
				updateDynamicBuffer(m_indexBuffers[ib->handle.idx], bx::strideAlign(_render->m_iboffset, 4), ib->data);
			}

			if (0 < _render->m_vboffset)
			{
				TransientVertexBuffer* vb = _render->m_transientVb;
				updateDynamicBuffer(m_vertexBuffers[vb->handle.idx], bx::strideAlign(_render->m_vboffset, 4), vb->data);
			}

			_render->sort();

			if (0 != (_render->m_mainSwapChain.flags & BGFX_SWAP_CHAIN_MSAA_MASK) )
			{
				traceOnce(Unsupported::Msaa, "MSAA asked of the window is ignored: BgfxView renders at sample count 1.");
			}

			if (0 != (m_reset & BGFX_RESET_CAPTURE) )
			{
				traceOnce(Unsupported::Capture, "G34: BGFX_RESET_CAPTURE calls no captureBegin or captureFrame (no read of the drawable inside a frame).");
			}

			if (0 != (_render->m_debug & BGFX_DEBUG_WIREFRAME) )
			{
				traceOnce(Unsupported::Wireframe, "G28: BGFX_DEBUG_WIREFRAME has no eacp fill mode.");
			}

			s_hiDpi = 0 != (_render->m_mainSwapChain.flags & BGFX_SWAP_CHAIN_HIDPI);

			if (NULL != s_frame)
			{
				updateWindowScale(*s_frame, _render->m_mainSwapChain);
			}

			if (NULL != s_frame
			&&  0 == (_render->m_debug & BGFX_DEBUG_IFH) )
			{
				submitItems(*s_frame, _render, _clearQuad, statsKeyType, statsNumPrimsRendered);
			}

			const int64_t timeEnd   = bx::getHPCounter();
			const int64_t frameTime = timeEnd - timeBegin;

			if (NULL != s_frame)
			{
				submitDebugText(*s_frame, _render, _textVideoMemBlitter, timeEnd, frameTime, statsKeyType, statsNumPrimsRendered);
			}

			endPass();
			m_lastPipelineKey = 0;
			m_lastPipeline    = NULL;

			Stats& perfStats = _render->m_perfStats;
			perfStats.cpuTimeBegin  = timeBegin;
			perfStats.cpuTimeEnd    = timeEnd;
			perfStats.cpuTimerFreq  = timerFreq;

			// eacp's timings are of the newest frame the GPU has finished, a few
			// behind this one (FrameTimings), in milliseconds; bgfx's are ticks
			// of gpuTimerFreq, here nanoseconds.
			const GPU::FrameTimings& timings = GPU::Device::shared().lastFrameTimings();
			perfStats.gpuTimeBegin  = 0;
			perfStats.gpuTimeEnd    = int64_t(timings.milliseconds*1000000.0);
			perfStats.gpuTimerFreq  = 1000000000;
			perfStats.gpuFrameNum   = uint32_t(timings.frameIndex);

			fillViewStats(_render, timings);

			perfStats.numDraw       = statsKeyType[0];
			perfStats.numCompute    = statsKeyType[1];
			perfStats.numBlit       = _render->m_numBlitItems;
			perfStats.numBlitRepack = _render->m_numBlitRepack;
			bx::memCopy(perfStats.numPrims, statsNumPrimsRendered, sizeof(perfStats.numPrims) );

			perfStats.gpuMemoryMax  = -INT64_MAX;
			perfStats.gpuMemoryUsed = -INT64_MAX;

			tracePerf(timeBegin, timeEnd, timerFreq);
		}

		/// In the off-screen capture mode (EACP_BGFX_CAPTURE, which shows no
		/// stats page), the CPU time of submit and the time from one submit to
		/// the next, averaged over each 100 frames, in the log.
		void tracePerf(int64_t _timeBegin, int64_t _timeEnd, int64_t _timerFreq)
		{
			if (!m_perfTrace)
			{
				return;
			}

			if (0 != m_perfLastBegin)
			{
				m_perfFrame += _timeBegin - m_perfLastBegin;
				m_perfSubmit += _timeEnd - _timeBegin;
				++m_perfNum;
			}

			m_perfLastBegin = _timeBegin;

			if (100 == m_perfNum)
			{
				const double toMs = 1000.0/double(_timerFreq)/double(m_perfNum);
				BX_TRACE("eacp perf: submit %.3f ms, frame %.3f ms (average of %d frames)"
					, double(m_perfSubmit)*toMs
					, double(m_perfFrame)*toMs
					, m_perfNum
					);
				m_perfFrame  = 0;
				m_perfSubmit = 0;
				m_perfNum    = 0;
			}
		}

		/// BGFX_DEBUG_PROFILER's per-view numbers: the CPU time of each view as
		/// submit walked it, and the GPU time of the passes labelled with that
		/// view's name (formatViewName starts with the view id) in eacp's last
		/// finished frame, laid end to end.
		void fillViewStats(Frame* _render, const GPU::FrameTimings& _timings)
		{
			Stats& perfStats = _render->m_perfStats;
			perfStats.numViews = 0;

			if (0 == (_render->m_debug & BGFX_DEBUG_PROFILER)
			||  NULL == perfStats.viewStats)
			{
				return;
			}

			const uint32_t numViews = bx::min<uint32_t>(m_numViewTimes, _render->m_numViewStats);
			int64_t gpuTime = 0;

			for (uint32_t ii = 0; ii < numViews; ++ii)
			{
				const ViewTime& time = m_viewTimes[ii];
				ViewStats& viewStats = perfStats.viewStats[ii];

				viewStats.view         = ViewId(time.m_view);
				viewStats.cpuTimeBegin = time.m_cpuBegin;
				viewStats.cpuTimeEnd   = time.m_cpuEnd;
				bx::strCopy(viewStats.name, BGFX_CONFIG_MAX_VIEW_NAME, getViewName(_render, time.m_view) );

				double milliseconds = 0.0;

				for (const GPU::PassTiming& pass : _timings.passes)
				{
					if (4 <= pass.label.size()
					&&  bx::isNumeric(pass.label[3])
					&&  uint32_t(atoi(pass.label.substr(0, 4).c_str() ) ) == time.m_view)
					{
						milliseconds += pass.milliseconds;
					}
				}

				viewStats.gpuTimeBegin = gpuTime;
				gpuTime += int64_t(milliseconds*1000000.0);
				viewStats.gpuTimeEnd   = gpuTime;
				viewStats.gpuFrameNum  = uint32_t(_timings.frameIndex);
			}

			perfStats.numViews = uint16_t(numViews);
		}

		static void updateDynamicBuffer(BufferEacp* _buffer, uint32_t _size, const void* _data)
		{
			if (NULL != _buffer)
			{
				_buffer->update(0, _size, _data);
			}
		}

		/// bgfx's backbuffer is the main swap chain's size, which the entry
		/// layer reports in points unless BGFX_RESET_HIDPI is set (as
		/// entry_osx.mm does); the eacp drawable is always in pixels, and eacp
		/// has no way to render it at a lower scale. So the backbuffer is mapped
		/// onto the drawable: view rects, scissors and the predefined
		/// u_viewRect/u_viewTexel of window draws are scaled by drawable over
		/// backbuffer, which on Retina without HIDPI is the backing scale (the
		/// picture stock Metal shows, rasterised at the panel's resolution) and
		/// with HIDPI is 1.
		void updateWindowScale(GPU::Frame& _frame, const SwapChain& _swapChain)
		{
			const ::eacp::Graphics::Point size = _frame.pixelSize();

			m_backbufferWidth  = uint16_t(bx::max<uint32_t>(_swapChain.width,  1) );
			m_backbufferHeight = uint16_t(bx::max<uint32_t>(_swapChain.height, 1) );
			m_windowScaleX     = 0.0f < size.x ? float(size.x)/float(m_backbufferWidth)  : 1.0f;
			m_windowScaleY     = 0.0f < size.y ? float(size.y)/float(m_backbufferHeight) : 1.0f;
		}

		/// A rect of the open pass's target from bgfx's units into the
		/// target's pixels: the window's are scaled (updateWindowScale), a
		/// texture's are its own.
		::eacp::Graphics::Rect toPassPixels(const Rect& _rect) const
		{
			return
			{
				float(_rect.m_x)*m_passScaleX,
				float(_rect.m_y)*m_passScaleY,
				float(_rect.m_width )*m_passScaleX,
				float(_rect.m_height)*m_passScaleY,
			};
		}

		/// A window view's rect in drawable pixels, for u_viewRect and
		/// u_viewTexel: what gl_FragCoord is measured in.
		Rect toWindowPixels(const Rect& _rect) const
		{
			const uint16_t x = uint16_t(float(_rect.m_x)*m_windowScaleX + 0.5f);
			const uint16_t y = uint16_t(float(_rect.m_y)*m_windowScaleY + 0.5f);
			const uint16_t r = uint16_t(float(_rect.m_x + _rect.m_width )*m_windowScaleX + 0.5f);
			const uint16_t b = uint16_t(float(_rect.m_y + _rect.m_height)*m_windowScaleY + 0.5f);
			return Rect(x, y, uint16_t(bx::max(r - x, 1) ), uint16_t(bx::max(b - y, 1) ) );
		}

		/// A pass on the window, continuing its depth when an earlier pass of
		/// this frame kept it and the descriptor does not clear.
		void beginWindowPass(GPU::Frame& _frame, GPU::RenderPassDescriptor& _descriptor, bool _clearDepth)
		{
			_descriptor.depthAction = !_clearDepth && m_windowDepthKept
				? GPU::DepthAction::Resume
				: GPU::DepthAction::Keep
				;

			m_computePass.reset();
			m_pass.reset();
			m_pass.emplace(PassMaker{ _frame, _descriptor, NULL });
			m_target          = windowTarget();
			m_passScaleX      = m_windowScaleX;
			m_passScaleY      = m_windowScaleY;
			m_passTexture     = kInvalidHandle;
			m_passDepth       = kInvalidHandle;
			m_passNumColor    = 0;
			m_windowDepthKept = true;
			m_needFlush       = true;
			m_lastPipelineKey = 0;
			m_lastPipeline    = NULL;
		}

		/// A pass into a render target texture, or into a framebuffer's colour
		/// attachments (_targets, each with its own clear, the first being
		/// _texture), its depth plane _depth's: _texture's own, or a depth
		/// texture's carrier the descriptor names (depthTarget, _depthIdx).
		/// The plane is kept at the end of every pass, so a later view on the
		/// same framebuffer that does not clear, or a draw sampling the
		/// depth, finds it.
		void beginTexturePass(GPU::Frame& _frame, TextureEacp& _texture, uint16_t _idx, TextureEacp& _depth, uint16_t _depthIdx, GPU::RenderPassDescriptor& _descriptor, bool _clearDepth, const FrameBufferEacp* _fb = NULL, std::span<const GPU::ColorTarget> _targets = {}, const uint16_t* _targetIdx = NULL, const FrameBufferEacp::Sub* _targetSub = NULL)
		{
			_descriptor.depthAction = !_clearDepth && _depth.m_depthKept
				? GPU::DepthAction::Resume
				: GPU::DepthAction::Keep
				;

			m_computePass.reset();
			m_pass.reset();

			m_pass.emplace(PassMaker{ _frame, _descriptor, &*_texture.m_texture, _targets });
			m_target            = textureTarget(_texture, _targets.empty() ? NULL : _fb);
			m_target.m_depth    = _depth.m_texture->hasDepth();
			m_target.m_stencil  = _depth.m_texture->hasStencil();
			m_passScaleX        = 1.0f;
			m_passScaleY        = 1.0f;
			m_passTexture       = _idx;
			m_passDepth         = _depthIdx;
			m_passNumColor      = 0;

			if (!_targets.empty() )
			{
				m_passNumColor = uint8_t(_targets.size() );

				for (uint8_t ii = 0; ii < m_passNumColor; ++ii)
				{
					m_passColor[ii] = _targetIdx[ii];
					m_passSub[ii]   = _targetSub[ii];
				}
			}
			_depth.m_depthKept = true;
			m_needFlush         = true;
			m_lastPipelineKey   = 0;
			m_lastPipeline      = NULL;
		}

		/// BGFX_ATTACHMENT_AUTO_GEN_MIPS: once the views on a framebuffer
		/// end (a view draws into something else, or the frame ends), each
		/// such colour attachment's texture gets its levels below 0 built
		/// from level 0, as renderer_mtl.cpp's FrameBufferMtl::resolve does
		/// when the next view binds another framebuffer. No pass may be open
		/// (Frame::generateMipmaps); a texture of one level, or of a format
		/// the device does not filter (no _MIP_AUTOGEN cap), is left.
		void resolveFrameBuffer(GPU::Frame& _frame)
		{
			const uint16_t fbh = m_genFrameBuffer;
			m_genFrameBuffer = kInvalidHandle;

			if (kInvalidHandle == fbh)
			{
				return;
			}

			const FrameBufferEacp& fb = m_frameBuffers[fbh];

			for (uint8_t ii = 0; ii < fb.m_num; ++ii)
			{
				const TextureEacp& texture = m_textures[fb.m_color[ii] ];

				if (!fb.m_autoGen[ii]
				||  !texture.isTarget()
				||  2 > texture.m_texture->mipLevels()
				||  0 == (g_caps.formats[texture.m_textureFormat] & BGFX_CAPS_FORMAT_TEXTURE_MIP_AUTOGEN) )
				{
					continue;
				}

				endPass();
				_frame.generateMipmaps(*texture.m_texture);
				m_needFlush = true;
			}
		}

		/// Ends whichever pass is open, render or compute: a command buffer
		/// takes one encoder at a time.
		void endPass()
		{
			m_computePass.reset();
			m_pass.reset();
			m_passTexture  = kInvalidHandle;
			m_passDepth    = kInvalidHandle;
			m_passNumColor = 0;
		}

		/// Whether the texture is one the open pass renders into, which a draw
		/// in it must not sample.
		bool isPassAttachment(uint16_t _idx) const
		{
			if (_idx == m_passTexture)
			{
				return true;
			}

			for (uint8_t ii = 0; ii < m_passNumColor; ++ii)
			{
				if (_idx == m_passColor[ii])
				{
					return true;
				}
			}

			return false;
		}

		// ---- blits ----------------------------------------------------------------

		/// A bgfx blit is a Frame::copy, recorded in order with the passes. No
		/// pass may be open when a copy is recorded, so whichever is open ends
		/// here and the next view's first draw opens another (its depth through
		/// DepthAction::Resume, as after a dispatch). bgfx asks that both
		/// textures of a blit have one format, which is what a copy asks, so
		/// every blit eacp can name is a copy and nothing is drawn: what is
		/// left out is what eacp cannot hold (a depth texture, G36; a level or
		/// slice the texture lacks), a volume to or from another kind, and more
		/// than one image between a buffer and a texture. A blit of several layers
		/// is one copy of a run of slices (sliceCount), of a volume's images
		/// one copy of a box (z and depth).
		void submitBlit(GPU::Frame& _frame, const BlitItem& _blit)
		{
			endPass();
			m_needFlush = true;

			if (_blit.m_src.isBuffer()
			&&  _blit.m_dst.isBuffer() )
			{
				BufferEacp* src = bufferOf(_blit.m_src);
				BufferEacp* dst = bufferOf(_blit.m_dst);

				if (NULL != src && src->m_static
				&&  NULL != dst && dst->m_static)
				{
					_frame.copy(*src->m_static, *dst->m_static, _blit.m_srcOffset, _blit.m_dstOffset, _blit.m_size);
				}

				return;
			}

			if (_blit.m_src.isBuffer()
			||  _blit.m_dst.isBuffer() )
			{
				const bool toBuffer = _blit.m_dst.isBuffer();
				BufferEacp*  buffer  = bufferOf(toBuffer ? _blit.m_dst : _blit.m_src);
				TextureEacp& texture = m_textures[toBuffer ? _blit.m_src.idx : _blit.m_dst.idx];

				const uint16_t z = toBuffer ? _blit.m_srcZ : _blit.m_dstZ;

				if (NULL == buffer
				||  !buffer->m_static
				||  !blitTexture(texture, toBuffer ? _blit.m_srcMip : _blit.m_dstMip, z, 1) )
				{
					return;
				}

				if (1 < _blit.m_depth)
				{
					traceOnce(Unsupported::BlitLevel, "a blit between a buffer and more than one layer or image of a texture copies the first (eacp's buffer-texture copies are one image).");
				}

				if (texture.isVolume()
				&&  0 != z)
				{
					traceOnce(Unsupported::BlitVolume, "a blit between a buffer and a 3D texture's image past the first is skipped (eacp's buffer-texture copies take no z).");
					return;
				}

				GPU::BufferTextureCopy layout;
				layout.bufferOffset = toBuffer ? _blit.m_dstOffset : _blit.m_srcOffset;
				layout.bytesPerRow  = int(_blit.m_rowPitch);
				layout.level        = toBuffer ? _blit.m_srcMip : _blit.m_dstMip;
				layout.slice        = texture.isVolume() ? 0 : int(z);
				layout.origin       = { int(toBuffer ? _blit.m_srcX : _blit.m_dstX), int(toBuffer ? _blit.m_srcY : _blit.m_dstY), texture.isVolume() ? int(z) : 0 };
				layout.size         = { int(_blit.m_width), int(_blit.m_height), 1 };

				if (toBuffer)
				{
					_frame.copy(*texture.m_texture, *buffer->m_static, layout);
				}
				else
				{
					_frame.copy(*buffer->m_static, *texture.m_texture, layout);
				}

				return;
			}

			TextureEacp& src = m_textures[_blit.m_src.idx];
			TextureEacp& dst = m_textures[_blit.m_dst.idx];

			if (!blitTexture(src, _blit.m_srcMip, _blit.m_srcZ, _blit.m_depth)
			||  !blitTexture(dst, _blit.m_dstMip, _blit.m_dstZ, _blit.m_depth) )
			{
				return;
			}

			// One copy: a volume's images are a box of z at slice 0 (both sides
			// volumes, eacp copying a volume only to or from one), anything
			// else's layers and faces a run of slices (sliceCount).
			const uint16_t depth = bx::max<uint16_t>(1, _blit.m_depth);

			if (src.isVolume() != dst.isVolume() )
			{
				traceOnce(Unsupported::BlitVolume, "a blit between a 3D texture and another kind is skipped (eacp copies a volume only to or from a volume).");
				return;
			}

			GPU::TextureCopyRegion region;
			region.sourceLevel       = _blit.m_srcMip;
			region.sourceSlice       = src.isVolume() ? 0 : _blit.m_srcZ;
			region.sourceOrigin      = { int(_blit.m_srcX), int(_blit.m_srcY), src.isVolume() ? int(_blit.m_srcZ) : 0 };
			region.destinationLevel  = _blit.m_dstMip;
			region.destinationSlice  = dst.isVolume() ? 0 : _blit.m_dstZ;
			region.destinationOrigin = { int(_blit.m_dstX), int(_blit.m_dstY), dst.isVolume() ? int(_blit.m_dstZ) : 0 };
			region.size              = { int(_blit.m_width), int(_blit.m_height), src.isVolume() ? int(depth) : 1 };
			region.sliceCount        = src.isVolume() ? 1 : int(depth);

			_frame.copy(*src.m_texture, *dst.m_texture, region);
		}

		/// Whether one side of a blit is a texture level eacp has: a colour
		/// texture (depth has no texture of its own, G36), a level it was made
		/// with, and the slices (a volume's images at that level) it has.
		bool blitTexture(const TextureEacp& _texture, uint8_t _mip, uint16_t _z, uint16_t _depth)
		{
			if (_texture.m_depthFormat)
			{
				traceOnce(Unsupported::BlitDepth, "G36: a blit to or from a depth texture is skipped (no eacp depth texture to copy).");
				return false;
			}

			if (!_texture.m_texture)
			{
				return false;
			}

			const uint32_t extent = _texture.isVolume()
				? mipExtent(_texture.m_depth, _mip)
				: _texture.slices()
				;

			if (_mip >= _texture.m_texture->mipLevels()
			||  uint32_t(_z) + bx::max<uint16_t>(1, _depth) > extent)
			{
				traceOnce(Unsupported::BlitLevel, "G6: a blit of a level, layer or image the texture was not made with (render targets have one level) is skipped.");
				return false;
			}

			return true;
		}

		/// The backend buffer behind a blit's or a read's buffer handle (bgfx
		/// hands over the vertex or index buffer under a dynamic one).
		BufferEacp* bufferOf(const Handle& _handle) const
		{
			if (Handle::IndexBuffer == _handle.type)
			{
				return m_indexBuffers[_handle.idx];
			}

			if (Handle::VertexBuffer   == _handle.type
			||  Handle::IndirectBuffer == _handle.type)
			{
				return m_vertexBuffers[_handle.idx];
			}

			return NULL;
		}

		void submitItems(GPU::Frame& _frame, Frame* _render, const ClearQuad& _clearQuad, uint32_t* _statsKeyType, uint32_t* _statsNumPrimsRendered)
		{
			static ViewState viewState;
			viewState.reset(_render);

			m_windowDepthKept = false;
			m_passTexture     = kInvalidHandle;
			m_passDepth       = kInvalidHandle;
			m_genFrameBuffer  = kInvalidHandle;
			m_passNumColor    = 0;
			m_numViewTimes    = 0;

			if (!m_fallbackDepth.m_depthKept
			&&  m_fallbackDepth.isTarget() )
			{
				GPU::RenderPassDescriptor descriptor;
				descriptor.clear = true;
				beginTexturePass(_frame, m_fallbackDepth, kInvalidHandle-1, m_fallbackDepth, kInvalidHandle, descriptor, true);
				endPass();
			}

			SortKey key;
			uint16_t view = UINT16_MAX;
			const View* renderView = &_render->view(0);
			char viewName[BGFX_CONFIG_MAX_VIEW_NAME] = "";

			UniformCacheState ucs(_render);
			BlitState bs(_render);

			bool viewSkipped = true;
			bool viewBegun   = false; // the view's first draw has opened (or joined) its pass
			TextureEacp* viewTarget = NULL;
			FrameBufferEacp* viewFrameBuffer = NULL;
			Rect viewScissorRect;
			viewScissorRect.clear();
			uint16_t currentScissor = UINT16_MAX - 1;
			uint32_t currentStencilRef = UINT32_MAX;

			// What the open pass has bound, so that an unchanged pipeline or
			// vertex stream is not sent again: eacp's setPipeline re-encodes
			// the depth-stencil state, winding and cull mode on every call.
			const PipelineEacp* currentPipeline = NULL;
			BoundStream currentStreams[BGFX_CONFIG_MAX_VERTEX_STREAMS+1];

			viewState.m_rect = _render->view(0).m_rect;

			for (uint32_t item = 0, numItems = _render->m_numRenderItems; item < numItems; ++item)
			{
				const bool isCompute = key.decode(_render->m_sortKeys[item], _render->m_viewRemap);
				_statsKeyType[isCompute]++;

				const uint32_t itemIdx       = _render->m_sortValues[item];
				const RenderItem& renderItem = _render->m_renderItem[itemIdx];

				if (key.m_view != view)
				{
					view = key.m_view;
					renderView = &_render->view(view);
					viewState.setView(*renderView);
					viewState.m_rect = renderView->m_rect;

					formatViewName(viewName, _render, view, "  ");

					submitUniformCache(ucs, view);

					while (bs.hasItem(view) )
					{
						submitBlit(_frame, bs.advance() );
					}

					noteViewTime(view);

					// A view's compute items get a pass, and a timing label, of
					// their own.
					m_computePass.reset();

					viewTarget      = NULL;
					viewFrameBuffer = NULL;
					viewSkipped     = false;
					viewBegun       = false;

					if (isValid(renderView->m_fbh) )
					{
						viewTarget      = frameBufferTarget(renderView->m_fbh.idx);
						viewFrameBuffer = &m_frameBuffers[renderView->m_fbh.idx];
						viewSkipped     = NULL == viewTarget;
					}
				}

				if (isCompute)
				{
					// A view's compute items sort before its draws; they need
					// no target, so a view whose framebuffer is skipped still
					// dispatches.
					submitCompute(_frame, viewState, view, *renderView, viewName, renderItem.compute, key.m_program, _render);
					continue;
				}

				if (viewSkipped)
				{
					continue;
				}

				if (!viewBegun
				||  !m_pass)
				{
					// The view's pass opens at its first draw, after its compute
					// items, and clears then (as renderer_mtl.cpp's does). A draw
					// whose pass a dispatch ended reopens it without the clear.
					beginView(_frame, _render, _clearQuad, *renderView, viewName, viewTarget, viewFrameBuffer, viewScissorRect, !viewBegun);
					viewBegun = true;

					// The view's draws see their rect in the pixels of what
					// they draw into (the view's compute items, before this,
					// saw bgfx's).
					viewState.m_rect = NULL == viewTarget
						? toWindowPixels(renderView->m_rect)
						: renderView->m_rect
						;

					currentScissor    = UINT16_MAX - 1;
					currentStencilRef = UINT32_MAX;
					currentPipeline   = NULL;
					m_lastPipelineKey = 0;
					m_lastPipeline    = NULL;

					for (BoundStream& bound : currentStreams)
					{
						bound = BoundStream();
					}
				}

				const RenderDraw& draw = renderItem.draw;

				if (isValid(draw.m_occlusionQuery) )
				{
					traceOnce(Unsupported::OcclusionQuery, "G10: an occlusion query never answers, and a draw conditional on one is drawn.");
				}

				if (_render->m_frameCache.isZeroArea(viewScissorRect, draw.m_scissor) )
				{
					continue;
				}

				if (0 == draw.m_streamMask
				||  UINT32_MAX == draw.m_streamMask
				||  !isValid(key.m_program) )
				{
					continue;
				}

				ProgramEacp* program = m_program[key.m_program.idx];

				if (NULL == program
				||  !program->m_library
				||  program->isCompute() )
				{
					continue;
				}

				const RenderBind& renderBind = _render->m_renderBind[draw.m_bindIdx];

				if (program->usesStorage()
				&&  hasImageBinding(renderBind) )
				{
					traceOnce(Unsupported::RenderImage, "G40: draws that bind a storage image to a render stage are skipped.");
					continue;
				}

				if (0 != program->m_vsh->m_numSamplers
				&&  !m_vertexTextures)
				{
					traceOnce(Unsupported::VertexTexture, "G17: draws whose vertex shader samples a texture are skipped (the device has no vertex-stage textures).");
					continue;
				}

				GPU::RenderPass& pass = *m_pass;

				// Streams, in the order bgfx's stream mask names them, compacted
				// to Metal buffers 0, 1, ...
				const VertexLayout* layouts[BGFX_CONFIG_MAX_VERTEX_STREAMS];
				GPU::BufferRange ranges[BGFX_CONFIG_MAX_VERTEX_STREAMS];
				uint8_t  numStreams  = 0;
				uint32_t numVertices = draw.m_numVertices;
				bool streamsValid    = true;

				for (BitMaskToIndexIteratorT it(draw.m_streamMask); !it.isDone(); it.next(), numStreams++)
				{
					const uint8_t idx = it.idx;
					const Stream& stream = draw.m_stream[idx];
					BufferEacp* vb = m_vertexBuffers[stream.m_handle.idx];

					if (NULL == vb)
					{
						streamsValid = false;
						break;
					}

					const uint16_t decl = isValid(stream.m_layoutHandle)
						? stream.m_layoutHandle.idx
						: vb->m_layoutHandle.idx
						;

					if (kInvalidHandle == decl)
					{
						streamsValid = false;
						break;
					}

					const VertexLayout& layout = m_vertexLayouts[decl];
					const uint32_t stride = bx::max<uint32_t>(layout.m_stride, 1);

					layouts[numStreams] = &layout;

					numVertices = bx::min(UINT32_MAX == draw.m_numVertices
						? vb->m_size/stride
						: draw.m_numVertices
						, numVertices
						);

					const uint32_t offset = stream.m_startVertex*stride;
					GPU::BufferRange range = vb->range(m_vertexStream);
					range.offset += offset;
					range.bytes   = bx::max<int64_t>(range.bytes - offset, 0);
					ranges[numStreams] = range;
				}

				if (!streamsValid)
				{
					continue;
				}

				const uint8_t numInstanceData = isValid(draw.m_instanceDataBuffer)
					? uint8_t(draw.m_instanceDataStride/16)
					: 0
					;

				const uint64_t state = draw.m_stateFlags;
				const PipelineEacp* pipeline = getPipeline(state, draw.m_stencil, key.m_program, numStreams, layouts, numInstanceData, draw.m_rgba);

				if (!pipeline->m_pipeline->isValid() )
				{
					continue;
				}

				if (pipeline != currentPipeline)
				{
					currentPipeline = pipeline;
					pass.setPipeline(*pipeline->m_pipeline);
				}

				for (uint8_t ii = 0; ii < numStreams; ++ii)
				{
					if (currentStreams[ii].set(ranges[ii]) )
					{
						pass.setVertexBuffer(ranges[ii], ii);
					}
				}

				if (0 < numInstanceData)
				{
					BufferEacp* inst = m_vertexBuffers[draw.m_instanceDataBuffer.idx];

					if (NULL == inst)
					{
						continue;
					}

					GPU::BufferRange range = inst->range(m_vertexStream);
					range.offset += draw.m_instanceDataOffset;
					range.bytes   = bx::max<int64_t>(range.bytes - draw.m_instanceDataOffset, 0);

					if (currentStreams[numStreams].set(range) )
					{
						pass.setVertexBuffer(range, numStreams);
					}
				}

				const uint32_t ref = (unpackStencil(0, draw.m_stencil)&BGFX_STENCIL_FUNC_REF_MASK)>>BGFX_STENCIL_FUNC_REF_SHIFT;
				if (ref != currentStencilRef)
				{
					currentStencilRef = ref;
					pass.setStencilReference(ref);
				}

				if (draw.m_scissor != currentScissor)
				{
					currentScissor = draw.m_scissor;

					Rect scissorRect = viewScissorRect;
					if (UINT16_MAX != draw.m_scissor)
					{
						scissorRect.setIntersect(viewScissorRect, _render->m_frameCache.m_rectCache.m_cache[draw.m_scissor]);
					}

					pass.setScissorRect(toPassPixels(scissorRect) );
				}

				if (UINT16_MAX != draw.m_depthBias
				||  0 != renderView->m_depthBias.m_constant
				||  0.0f != renderView->m_depthBias.m_slopeScale)
				{
					traceOnce(Unsupported::DepthBias, "G27: depth bias has no eacp equivalent and is ignored.");
				}

				viewState.m_alphaRef = ( (state&BGFX_STATE_ALPHA_REF_MASK)>>BGFX_STATE_ALPHA_REF_SHIFT)/255.0f;

				rendererUpdateUniforms(this, _render->m_uniformBuffer[draw.m_uniformIdx], draw.m_uniformBegin, draw.m_uniformEnd);

				if (NULL != program->m_vsh->m_constantBuffer)
				{
					commit(*program->m_vsh->m_constantBuffer);
				}

				if (NULL != program->m_fsh
				&&  NULL != program->m_fsh->m_constantBuffer)
				{
					commit(*program->m_fsh->m_constantBuffer);
				}

				viewState.setPredefined<4>(this, view, *renderView, *program, _render, draw);

				bindUniforms(pass, *program);

				// A stage is one slot for the whole program in bgfx, so a
				// texture both shaders name is bound to each of them at the
				// same index (Metal's vertex and fragment index spaces are
				// separate; D3D12 and Vulkan share it, and it is the same bind).
				bindTextures(pass, *program->m_vsh, renderBind, true);

				if (NULL != program->m_fsh)
				{
					bindTextures(pass, *program->m_fsh, renderBind, false);
				}

				if (program->usesStorage() )
				{
					bindRenderStorage(pass, renderBind);
				}

				if (isValid(draw.m_indirectBuffer) )
				{
					submitIndirect(pass, draw);
					continue;
				}

				const uint8_t primIndex = uint8_t( (state&BGFX_STATE_PT_MASK)>>BGFX_STATE_PT_SHIFT);
				const PrimInfo& prim = s_primInfo[primIndex];
				const uint32_t numInstances = bx::max<uint32_t>(draw.m_numInstances, 1);
				uint32_t numPrimsSubmitted = 0;

				if (isValid(draw.m_indexBuffer) )
				{
					BufferEacp* ib = m_indexBuffers[draw.m_indexBuffer.idx];

					if (NULL == ib)
					{
						continue;
					}

					const bool isIndex16 = draw.isIndex16();
					const uint32_t indexSize = isIndex16 ? 2 : 4;
					const GPU::IndexFormat format = isIndex16 ? GPU::IndexFormat::UInt16 : GPU::IndexFormat::UInt32;

					uint32_t numIndices = draw.m_numIndices;
					uint32_t startIndex = draw.m_startIndex;

					if (UINT32_MAX == numIndices)
					{
						numIndices = ib->m_size/indexSize;
						startIndex = 0;
					}

					if (numIndices < prim.m_min)
					{
						continue;
					}

					const GPU::BufferRange range = ib->range(m_indexStream);

					if (1 < numInstances)
					{
						pass.drawIndexedInstanced(range, int(numIndices), int(numInstances), format, int(startIndex) );
					}
					else
					{
						pass.drawIndexed(range, int(numIndices), format, int(startIndex) );
					}

					numPrimsSubmitted = numIndices/prim.m_div - prim.m_sub;
				}
				else
				{
					if (numVertices < prim.m_min)
					{
						continue;
					}

					if (1 < numInstances)
					{
						pass.drawInstanced(int(numVertices), int(numInstances) );
					}
					else
					{
						pass.draw(int(numVertices) );
					}

					numPrimsSubmitted = numVertices/prim.m_div - prim.m_sub;
				}

				_statsNumPrimsRendered[primIndex] += numPrimsSubmitted*numInstances;
			}

			resolveFrameBuffer(_frame);

			while (bs.hasItem(BGFX_CONFIG_MAX_VIEWS) )
			{
				submitBlit(_frame, bs.advance() );
			}

			noteViewTime(UINT16_MAX);
		}

		/// An indirect draw, everything but the draw itself set up as for a
		/// direct one. bgfx's records are BGFX_CONFIG_DRAW_INDIRECT_STRIDE bytes
		/// apart and begin with eacp's DrawArguments or DrawIndexedArguments;
		/// each record's firstIndex counts from the start of the index buffer.
		/// The count form, with the number of draws in a uint32 of an index
		/// buffer, is reached only where BGFX_CAPS_DRAW_INDIRECT_COUNT is set.
		/// The primitives drawn are on the GPU, so the stats do not count them.
		void submitIndirect(GPU::RenderPass& _pass, const RenderDraw& _draw)
		{
			BufferEacp* args = m_vertexBuffers[_draw.m_indirectBuffer.idx];

			if (NULL == args
			||  !args->m_static)
			{
				return;
			}

			const uint32_t numRecords = args->m_size/BGFX_CONFIG_DRAW_INDIRECT_STRIDE;

			if (_draw.m_startIndirect >= numRecords)
			{
				return;
			}

			const uint32_t numDrawIndirect = bx::min(numRecords - _draw.m_startIndirect, _draw.m_numIndirect);
			const int64_t  offset          = int64_t(_draw.m_startIndirect)*BGFX_CONFIG_DRAW_INDIRECT_STRIDE;
			const int64_t  stride          = BGFX_CONFIG_DRAW_INDIRECT_STRIDE;

			const GPU::Buffer* count = NULL;
			int64_t countOffset      = 0;

			if (isValid(_draw.m_numIndirectBuffer) )
			{
				BufferEacp* num = m_indexBuffers[_draw.m_numIndirectBuffer.idx];

				if (NULL == num
				||  !num->m_static)
				{
					traceOnce(Unsupported::Indirect, "an indirect draw count in a buffer no kernel writes is skipped.");
					return;
				}

				count       = &*num->m_static;
				countOffset = int64_t(_draw.m_numIndirectIndex)*sizeof(uint32_t);
			}

			if (isValid(_draw.m_indexBuffer) )
			{
				BufferEacp* ib = m_indexBuffers[_draw.m_indexBuffer.idx];

				if (NULL == ib)
				{
					return;
				}

				const GPU::IndexFormat format = _draw.isIndex16() ? GPU::IndexFormat::UInt16 : GPU::IndexFormat::UInt32;
				const GPU::BufferRange range  = ib->range(m_indexStream);

				if (NULL != count)
				{
					_pass.drawIndexedIndirectCount(range, format, *args->m_static, offset, *count, countOffset, int(numDrawIndirect), stride);
				}
				else
				{
					_pass.drawIndexedIndirect(range, format, *args->m_static, offset, int(numDrawIndirect), stride);
				}
			}
			else if (NULL != count)
			{
				_pass.drawIndirectCount(*args->m_static, offset, *count, countOffset, int(numDrawIndirect), stride);
			}
			else
			{
				_pass.drawIndirect(*args->m_static, offset, int(numDrawIndirect), stride);
			}
		}

		/// CPU time per view for BGFX_DEBUG_PROFILER, as the views are walked;
		/// the GPU half is filled from eacp's pass timings at the end of submit.
		void noteViewTime(uint16_t _view)
		{
			const int64_t now = bx::getHPCounter();

			if (0 < m_numViewTimes)
			{
				m_viewTimes[m_numViewTimes-1].m_cpuEnd = now;
			}

			if (UINT16_MAX != _view
			&&  m_numViewTimes < BX_COUNTOF(m_viewTimes) )
			{
				ViewTime& time = m_viewTimes[m_numViewTimes++];
				time.m_view     = _view;
				time.m_cpuBegin = now;
				time.m_cpuEnd   = now;
			}
		}

		/// A view on the window or on a texture framebuffer: a new pass when
		/// none is open on that target or when the view clears the whole target
		/// (the pass's own clear), otherwise the open pass carries on and a
		/// partial clear is the clear quad. A texture target's rects are in its
		/// own pixels.
		void beginView(GPU::Frame& _frame, Frame* _render, const ClearQuad& _clearQuad, const View& _view, const char* _viewName, TextureEacp* _target, FrameBufferEacp* _fb, Rect& _viewScissorRect, bool _clear)
		{
			Clear clear = _view.m_clear;

			if (!_clear)
			{
				clear.m_flags = BGFX_CLEAR_NONE;
			}

			const Rect& viewRect = _view.m_rect;

			const uint16_t targetIdx = NULL == _target ? kInvalidHandle : uint16_t(_target - m_textures);
			const uint16_t fbIdx     = NULL == _fb     ? kInvalidHandle : uint16_t(_fb - m_frameBuffers);

			if (fbIdx != m_genFrameBuffer)
			{
				// The views on the last framebuffer are over: its mips.
				resolveFrameBuffer(_frame);
				m_genFrameBuffer = fbIdx;
			}

			Rect targetRect;

			if (NULL == _target)
			{
				targetRect = Rect(0, 0, m_backbufferWidth, m_backbufferHeight);
			}
			else
			{
				// The level the framebuffer renders into.
				const uint8_t level = NULL != _fb ? _fb->targetSub().m_level : 0;
				targetRect = Rect(0, 0, uint16_t(mipExtent(uint32_t(_target->m_texture->width() ), level) ), uint16_t(mipExtent(uint32_t(_target->m_texture->height() ), level) ) );
			}

			Rect clippedRect;
			clippedRect.setIntersect(_view.m_clippedRect, targetRect);

			Rect scissorRect;
			scissorRect.setIntersect(_view.m_scissor, targetRect);

			_viewScissorRect = _view.m_scissor.isZero() ? clippedRect : scissorRect;

			const uint16_t flags = clear.m_flags & BGFX_CLEAR_MASK;
			const bool fullRect = true
				&& 0 == viewRect.m_x
				&& 0 == viewRect.m_y
				&& viewRect.m_width  >= targetRect.m_width
				&& viewRect.m_height >= targetRect.m_height
				;

			const bool clearsColor        = 0 != (flags & BGFX_CLEAR_COLOR);
			const bool clearsDepthStencil = 0 != (flags & (BGFX_CLEAR_DEPTH|BGFX_CLEAR_STENCIL) );
			// The texture whose depth plane the pass attaches: the depth
			// texture's own carrier (depthTarget), or the target's.
			const uint16_t depthIdx = NULL != _fb && _fb->m_ownDepth ? _fb->m_depth : kInvalidHandle;
			TextureEacp* depthHolder = kInvalidHandle != depthIdx
				? &m_textures[depthIdx]
				: _target
				;

			const bool hasStencil = NULL == _target ? windowHasStencil() : depthHolder->m_texture->hasStencil();

			// G29: a pass clears depth and stencil together, depth to the far
			// plane. A clear of one plane only (with a stencil plane there to
			// keep), or of depth to another value, is the clear quad's, which
			// writes exactly the planes and the value asked for.
			const bool splitClear = true
				&& hasStencil
				&& clearsDepthStencil
				&& (BGFX_CLEAR_DEPTH|BGFX_CLEAR_STENCIL) != (flags & (BGFX_CLEAR_DEPTH|BGFX_CLEAR_STENCIL) )
				;
			const bool otherDepth = 0 != (flags & BGFX_CLEAR_DEPTH) && 1.0f != clear.m_depth;
			const bool passClearsDepth = fullRect && clearsDepthStencil && !splitClear && !otherDepth;

			uint16_t quadFlags = flags;

			// A framebuffer of several colour attachments, or of an image other
			// than level 0, slice 0 of its texture, is one pass over
			// ColorTargets (Frame::beginMultiTargetPass), and the same pass
			// only while the attachments and their images are the same. A
			// depth-only one at a layer is its carrier at that layer.
			uint8_t numColor = 0;
			uint16_t passIdx[BGFX_CONFIG_MAX_FRAME_BUFFER_ATTACHMENTS];
			FrameBufferEacp::Sub passSub[BGFX_CONFIG_MAX_FRAME_BUFFER_ATTACHMENTS] = {};

			if (NULL != _fb
			&&  _fb->needsTargets() )
			{
				if (0 == _fb->m_num)
				{
					passIdx[0] = _fb->m_target;
					passSub[0] = _fb->m_depthSub;
					numColor   = 1;
				}
				else
				{
					numColor = _fb->m_num;

					for (uint8_t ii = 0; ii < numColor; ++ii)
					{
						passIdx[ii] = _fb->m_color[ii];
						passSub[ii] = _fb->m_sub[ii];
					}
				}
			}

			bool sameColors = numColor == m_passNumColor;

			for (uint8_t ii = 0; ii < numColor && sameColors; ++ii)
			{
				sameColors = passIdx[ii] == m_passColor[ii] && passSub[ii] == m_passSub[ii];
			}
			const bool samePass = m_pass && m_passTexture == targetIdx && m_passDepth == depthIdx && sameColors;

			if (!samePass
			|| (fullRect && BGFX_CLEAR_NONE != flags) )
			{
				GPU::RenderPassDescriptor descriptor;
				descriptor.label = _viewName;
				descriptor.clear = false;

				const bool palette = 0 != (clear.m_flags & BGFX_CLEAR_COLOR_USE_PALETTE);

				if (fullRect && clearsColor)
				{
					// A palette index of UINT8_MAX leaves that attachment as it
					// is (Clear::getColorSkipMask).
					descriptor.clear      = !palette || UINT8_MAX != clear.m_index[0] || 0 != numColor;
					descriptor.clearColor = clearColor(clear, _render->m_colorPalette);
					quadFlags &= ~BGFX_CLEAR_COLOR;
				}

				// Each attachment its own clear: bgfx's palette gives one colour
				// per attachment (clear.m_index[n]), and skips the ones at
				// UINT8_MAX.
				GPU::ColorTarget targets[BGFX_CONFIG_MAX_FRAME_BUFFER_ATTACHMENTS];

				for (uint8_t ii = 0; ii < numColor; ++ii)
				{
					targets[ii].texture    = &*m_textures[passIdx[ii] ].m_texture;
					targets[ii].clear      = descriptor.clear && (!palette || UINT8_MAX != clear.m_index[ii]);
					targets[ii].clearColor = clearColor(clear, _render->m_colorPalette, ii);
					targets[ii].level      = int(passSub[ii].m_level);
					targets[ii].slice      = int(passSub[ii].m_slice);
					targets[ii].layerCount = int(passSub[ii].m_layers);
				}

				if (kInvalidHandle != depthIdx)
				{
					// The depth texture's own plane at the depth attachment's
					// image (a layered pass covering colour 0's run of it).
					descriptor.depthTarget.texture = &*depthHolder->m_texture;
					descriptor.depthTarget.level   = int(_fb->m_depthSub.m_level);
					descriptor.depthTarget.slice   = int(_fb->m_depthSub.m_slice);
				}

				if (passClearsDepth)
				{
					descriptor.clearStencil = clear.m_stencil;
					quadFlags &= ~(BGFX_CLEAR_DEPTH|BGFX_CLEAR_STENCIL);
				}

				// BGFX_CLEAR_DISCARD_* asks nothing of eacp: every plane is
				// stored, so a later view can load it.

				// A plane not cleared here picks up what the last pass on the
				// target kept, or starts cleared when none has.
				if (NULL == _target)
				{
					beginWindowPass(_frame, descriptor, passClearsDepth);
				}
				else
				{
					beginTexturePass(_frame, *_target, targetIdx, *depthHolder, depthIdx, descriptor, passClearsDepth, _fb, std::span<const GPU::ColorTarget>(targets, numColor), passIdx, passSub);
				}
			}

			GPU::RenderPass& pass = *m_pass;

			if (viewRect.m_x + viewRect.m_width  <= targetRect.m_width
			&&  viewRect.m_y + viewRect.m_height <= targetRect.m_height
			&&  0 < viewRect.m_width
			&&  0 < viewRect.m_height)
			{
				pass.setViewport(
					  toPassPixels(viewRect)
					, _view.m_minDepth
					, _view.m_maxDepth
					);
			}
			else
			{
				// G30: eacp rejects a viewport past the target, which Metal and
				// D3D12 both take. The clipped rect is the nearest thing.
				traceOnce(Unsupported::Viewport, "G30: a view rect past the target is drawn through the clipped rect.");
				pass.setViewport(
					  toPassPixels(clippedRect)
					, _view.m_minDepth
					, _view.m_maxDepth
					);
			}

			pass.setScissorRect(toPassPixels(clippedRect) );

			if (BGFX_CLEAR_NONE != quadFlags)
			{
				Clear quadClear = clear;
				quadClear.m_flags = quadFlags;
				clearQuad(pass, _clearQuad, quadClear, _render->m_colorPalette);
			}
		}

		/// The stats page (BGFX_DEBUG_STATS) or the examples' debug text
		/// (BGFX_DEBUG_TEXT) over the window, as renderer_mtl.cpp's end of
		/// submit, in the frame's last pass or a new one that loads it.
		void submitDebugText(GPU::Frame& _frame, Frame* _render, TextVideoMemBlitter& _blitter, int64_t _timeEnd, int64_t _frameTime, const uint32_t* _statsKeyType, const uint32_t* _statsNumPrimsRendered)
		{
			const bool stats = 0 != (_render->m_debug & (BGFX_DEBUG_IFH|BGFX_DEBUG_STATS) );
			const bool text  = 0 != (_render->m_debug & BGFX_DEBUG_TEXT);

			static int64_t min = _frameTime;
			static int64_t max = _frameTime;
			min = bx::min<int64_t>(min, _frameTime);
			max = bx::max<int64_t>(max, _frameTime);

			if (!stats
			&&  !text)
			{
				return;
			}

			if (!m_pass
			||  kInvalidHandle != m_passTexture)
			{
				GPU::RenderPassDescriptor descriptor;
				descriptor.label = "debugtext";
				descriptor.clear = false;
				beginWindowPass(_frame, descriptor, false);
			}

			if (stats)
			{
				TextVideoMem& tvm = m_textVideoMem;
				tvm.resize(false, m_backbufferWidth, m_backbufferHeight);

				static int64_t next = _timeEnd;
				const int64_t timerFreq = bx::getHPFrequency();

				if (_timeEnd >= next)
				{
					next = _timeEnd + timerFreq;

					const double freq = double(timerFreq);
					const double toMs = 1000.0/freq;

					tvm.clear();
					uint16_t pos = 0;
					tvm.printf(0, pos++, BGFX_CONFIG_DEBUG ? 0x8c : 0x8f
						, " %s (%s) / " BX_COMPILER_NAME " / " BX_CPU_NAME " / " BX_ARCH_NAME " / " BX_PLATFORM_NAME
						  " / Version 1.%d.%d (commit: " BGFX_REV_SHA1 ")"
						, getRendererName()
						, GPU::Device::shared().name().c_str()
						, BGFX_API_VERSION
						, BGFX_REV_NUMBER
						);

					pos = 10;
					tvm.printf(10, pos++, 0x8b, "        Frame: %7.3f, % 7.3f \x1f, % 7.3f \x1e [ms] / % 6.2f FPS "
						, double(_frameTime)*toMs
						, double(min)*toMs
						, double(max)*toMs
						, freq/_frameTime
						);

					tvm.printf(10, pos++, 0x8b, "    Submitted: %5d (draw %5d, compute %4d) / Binds: %4d / CPU %3.4f [ms] / GPU %3.4f [ms]"
						, _render->m_numRenderItems
						, _statsKeyType[0]
						, _statsKeyType[1]
						, _render->m_numRenderBinds
						, double(_frameTime)*toMs
						, GPU::Device::shared().lastFrameTimings().milliseconds
						);

					for (uint32_t ii = 0; ii < Topology::Count; ++ii)
					{
						tvm.printf(10, pos++, 0x8b, "   %10s: %7d"
							, getName(Topology::Enum(ii) )
							, _statsNumPrimsRendered[ii]
							);
					}

					tvm.printf(10, pos++, 0x8b, "     DVB size: %7d ", _render->m_vboffset);
					tvm.printf(10, pos++, 0x8b, "     DIB size: %7d ", _render->m_iboffset);

					pos++;
					uint8_t attr[2] = { 0x8c, 0x8a };
					uint8_t attrIndex = _render->m_waitSubmit < _render->m_waitRender;

					tvm.printf(10, pos++, attr[attrIndex    &1], " Submit wait: %3.4f [ms]", _render->m_waitSubmit*toMs);
					tvm.printf(10, pos++, attr[(attrIndex+1)&1], " Render wait: %3.4f [ms]", _render->m_waitRender*toMs);

					min = _frameTime;
					max = _frameTime;
				}

				dbgTextSubmit(this, _blitter, tvm, _render->m_debugFrameBuffer, _render->m_debugTextScale);
			}
			else
			{
				dbgTextSubmit(this, _blitter, _render->m_textVideoMem, _render->m_debugFrameBuffer, _render->m_debugTextScale);
			}
		}

		void dbgTextRenderBegin(TextVideoMemBlitter& /*_blitter*/, FrameBufferHandle _handle) override
		{
			if (isValid(_handle) )
			{
				traceOnce(Unsupported::DebugTextTarget, "debug text into a texture framebuffer (setDebug's framebuffer) is drawn on the window.");
			}
		}

		/// One batch of the debug text: a transient VB/IB draw over the whole
		/// target with a pixel ortho, the font in the blitter's R8 texture.
		void dbgTextRender(TextVideoMemBlitter& _blitter, uint32_t _numIndices) override
		{
			const uint32_t numVertices = _numIndices*4/6;

			BufferEacp* ib = m_indexBuffers[_blitter.m_ib->handle.idx];
			BufferEacp* vb = m_vertexBuffers[_blitter.m_vb->handle.idx];
			ProgramEacp* program = isValid(_blitter.m_program) ? m_program[_blitter.m_program.idx] : NULL;
			const TextureEacp& texture = m_textures[_blitter.m_texture.idx];

			if (0 == numVertices
			||  !m_pass
			||  NULL == ib
			||  NULL == vb
			||  NULL == program
			||  !program->m_library
			||  !texture.m_texture)
			{
				return;
			}

			ib->update(0, bx::strideAlign(_numIndices*2, 4), _blitter.m_ib->data);
			vb->update(0, numVertices*_blitter.m_layout.m_stride, _blitter.m_vb->data);

			const uint64_t state = 0
				| BGFX_STATE_WRITE_RGB
				| BGFX_STATE_WRITE_A
				| BGFX_STATE_DEPTH_TEST_ALWAYS
				;

			const VertexLayout* layout = &_blitter.m_layout;
			const PipelineEacp* pipeline = getPipeline(state, 0, _blitter.m_program, 1, &layout, 0);

			if (!pipeline->m_pipeline->isValid() )
			{
				return;
			}

			// The text is laid out on bgfx's backbuffer, which covers the whole
			// drawable (updateWindowScale).
			GPU::RenderPass& pass = *m_pass;
			const ::eacp::Graphics::Point size = s_frame->pixelSize();
			const float width  = float(m_backbufferWidth);
			const float height = float(m_backbufferHeight);

			pass.setViewport({ 0.0f, 0.0f, float(size.x), float(size.y) }, 0.0f, 1.0f);
			pass.setScissorRect({ 0.0f, 0.0f, float(size.x), float(size.y) });
			pass.setPipeline(*pipeline->m_pipeline);

			float proj[16];
			bx::mtxOrtho(proj, 0.0f, width, height, 0.0f, 0.0f, 1000.0f, 0.0f, false);

			for (uint8_t ii = 0; ii < program->m_numPredefined; ++ii)
			{
				const PredefinedUniform& predefined = program->m_predefined[ii];

				// vs_debugfont reads u_modelViewProj; renderer_mtl.cpp writes
				// the ortho into whichever predefined the program has.
				const uint8_t type = predefined.m_type & ~kUniformFragmentBit;

				if (PredefinedUniform::ViewProj      == type
				||  PredefinedUniform::ModelViewProj == type)
				{
					setShaderUniform(predefined.m_type, predefined.m_loc, proj, 4);
				}
			}

			bindUniforms(pass, *program);

			pass.setFragmentTexture(*texture.m_texture, 0, getSampler(uint32_t(texture.m_flags), 0, UINT8_MAX) );
			pass.setVertexBuffer(vb->range(m_vertexStream), 0);
			pass.drawIndexed(ib->range(m_indexStream), int(_numIndices), GPU::IndexFormat::UInt16, 0);

			m_lastPipelineKey = 0;
			m_lastPipeline    = NULL;
		}

		void dbgTextRenderEnd(TextVideoMemBlitter& /*_blitter*/) override
		{
		}

		GPU::GPUView* m_view;

		BufferEacp*  m_indexBuffers[BGFX_CONFIG_MAX_INDEX_BUFFERS];
		BufferEacp*  m_vertexBuffers[BGFX_CONFIG_MAX_VERTEX_BUFFERS];
		ShaderEacp*  m_shaders[BGFX_CONFIG_MAX_SHADERS];
		ProgramEacp* m_program[BGFX_CONFIG_MAX_PROGRAMS];
		VertexLayout m_vertexLayouts[BGFX_CONFIG_MAX_VERTEX_LAYOUTS];

		std::unordered_map<uint32_t, PipelineEacp*> m_pipelines;
		uint32_t m_lastPipelineKey = 0;
		const PipelineEacp* m_lastPipeline = NULL;

		struct PipelineQuery
		{
			uint64_t m_state;
			uint64_t m_stencil;
			uint32_t m_layoutHash[BGFX_CONFIG_MAX_VERTEX_STREAMS];
			uint16_t m_program;
			uint8_t  m_numStreams;
			uint8_t  m_numInstanceData;
			uint32_t m_rgba;
			uint8_t  m_mrtMask;
		};

		PipelineQuery m_lastQuery = {}; // what m_lastPipeline was looked up for

		TextureEacp m_textures[BGFX_CONFIG_MAX_TEXTURES];
		TextureEacp m_fallback[5]; // by GPU::TextureType: 2D, cube, 2D array, cube array, 3D
		TextureEacp m_fallbackDepth;
		FrameBufferEacp m_frameBuffers[BGFX_CONFIG_MAX_FRAME_BUFFERS];
		std::unordered_map<uint64_t, GPU::Sampler> m_samplers;
		const float (*m_colorPalette)[4] = NULL;
		uint32_t m_reset = 0;

		TextVideoMem m_textVideoMem;

		GPU::StreamingBuffers m_vertexStream;
		GPU::StreamingBuffers m_indexStream;

		std::optional<GPU::RenderPass> m_pass;
		std::optional<GPU::ComputePass> m_computePass; // the open run of compute items, one view's
		Target   m_target;
		uint16_t m_passTexture     = kInvalidHandle; // the open pass's texture target, or the window
		uint16_t m_passColor[BGFX_CONFIG_MAX_FRAME_BUFFER_ATTACHMENTS] = {}; // ... and its colour attachments, when several
		uint8_t  m_passNumColor    = 0;
		FrameBufferEacp::Sub m_passSub[BGFX_CONFIG_MAX_FRAME_BUFFER_ATTACHMENTS]; // ... and the image of each
		uint16_t m_passDepth       = kInvalidHandle; // the depth texture whose own plane the open pass attaches (depthTarget)
		uint16_t m_genFrameBuffer  = kInvalidHandle; // the framebuffer the last texture view drew into, whose mips are built when its views end
		bool     m_windowDepthKept = false;          // a window pass this frame kept the depth plane
		bool     m_needFlush       = false;          // passes recorded since the last Frame::flush
		float    m_passScaleX       = 1.0f;           // the open pass's target pixels per bgfx unit
		float    m_passScaleY       = 1.0f;
		float    m_windowScaleX     = 1.0f;           // drawable pixels per backbuffer unit
		float    m_windowScaleY     = 1.0f;
		uint16_t m_backbufferWidth  = 1;              // the main swap chain, in the entry layer's units
		uint16_t m_backbufferHeight = 1;

		struct ViewTime
		{
			uint16_t m_view;
			int64_t  m_cpuBegin;
			int64_t  m_cpuEnd;
		};

		ViewTime m_viewTimes[BGFX_CONFIG_MAX_VIEWS];
		uint32_t m_numViewTimes = 0;

		bool    m_perfTrace     = false;
		bool    m_vertexTextures = false;
		int64_t m_perfLastBegin = 0;
		int64_t m_perfFrame     = 0;
		int64_t m_perfSubmit    = 0;
		int32_t m_perfNum       = 0;

		alignas(16) uint8_t m_vsScratch[64<<10];
		alignas(16) uint8_t m_fsScratch[64<<10];
	};

	static RendererContextEacp* s_renderEacp;

	RendererContextI* rendererCreate(const Init& _init)
	{
		s_renderEacp = BX_NEW(g_allocator, RendererContextEacp);
		if (!s_renderEacp->init(_init) )
		{
			bx::deleteObject(g_allocator, s_renderEacp);
			s_renderEacp = NULL;
		}
		return s_renderEacp;
	}

	void rendererDestroy()
	{
		bx::deleteObject(g_allocator, s_renderEacp);
		s_renderEacp = NULL;
	}

	RenderFrame::Enum renderFrame(GPU::Frame& _frame, int32_t _msecs)
	{
		s_frame = &_frame;
		const RenderFrame::Enum result = bgfx::renderFrame(_msecs);
		s_frame = NULL;
		return result;
	}

	bool isHiDpi()
	{
		return s_hiDpi;
	}

} /* namespace eacp */ } // namespace bgfx

#else

namespace bgfx { namespace eacp
{
	RendererContextI* rendererCreate(const Init& _init)
	{
		BX_UNUSED(_init);
		return NULL;
	}

	void rendererDestroy()
	{
	}
} /* namespace eacp */ } // namespace bgfx

#endif // BGFX_CONFIG_RENDERER_EACP
