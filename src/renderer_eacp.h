/*
 * Copyright 2011-2026 Branimir Karadzic. All rights reserved.
 * License: https://github.com/bkaradzic/bgfx/blob/master/LICENSE
 */

#ifndef BGFX_RENDERER_EACP_H_HEADER_GUARD
#define BGFX_RENDERER_EACP_H_HEADER_GUARD

#include "renderer.h"

#include <bgfx/eacp.h>

#include <eacp/Core/Threads/ThreadUtils.h>
#include <eacp/GPU/Device/Device.h>
#include <eacp/GPU/Frame/Frame.h>
#include <eacp/GPU/View/GPUView.h>

namespace bgfx { namespace eacp
{
	namespace GPU = ::eacp::GPU;

	/// The frame `bgfx::eacp::renderFrame` made current, or NULL outside it.
	/// Only `submit` draws, and only into this.
	GPU::Frame* currentFrame();

} /* namespace eacp */ } // namespace bgfx

#endif // BGFX_RENDERER_EACP_H_HEADER_GUARD
