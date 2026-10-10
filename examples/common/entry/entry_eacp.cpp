/*
 * Copyright 2011-2026 Branimir Karadzic. All rights reserved.
 * License: https://github.com/bkaradzic/bgfx/blob/master/LICENSE
 */

#include "entry_p.h"

#if ENTRY_CONFIG_USE_EACP

#include <bgfx/bgfx.h>
#include <bgfx/eacp.h>

#include <bx/thread.h>

#include <eacp/Core/App/App.h>
#include <eacp/Core/Threads/EventLoop.h>
#include <eacp/GPU/Device/Device.h>
#include <eacp/GPU/Frame/Frame.h>
#include <eacp/GPU/View/GPUView.h>
#include <eacp/Graphics/Window/Window.h>

#include <atomic>
#include <cstdlib>
#include <optional>
#include <string>

namespace entry
{
	namespace Graphics = ::eacp::Graphics;
	namespace GPU      = ::eacp::GPU;
	namespace Apps     = ::eacp::Apps;
	namespace Threads  = ::eacp::Threads;

	// GPUView::render runs on the main thread with a drawable already acquired,
	// so an unbounded wait for the API thread's frame would hold the drawable
	// and the event loop hostage. A timeout presents an undrawn drawable for
	// that tick instead and tries again on the next one.
	static constexpr int32_t kRenderFrameTimeoutMs = 100;

	struct MainThreadEntry
	{
		int m_argc = 0;
		const char* const* m_argv = NULL;
		std::atomic<bool> m_done = false;

		static int32_t threadFunc(bx::Thread* _thread, void* _userData)
		{
			BX_UNUSED(_thread);

			MainThreadEntry* self = (MainThreadEntry*)_userData;
			const int32_t result = main(self->m_argc, self->m_argv);
			self->m_done = true;
			return result;
		}
	};

	static Key::Enum translateKey(uint16_t _keyCode)
	{
		namespace KeyCode = Graphics::KeyCode;

		switch (_keyCode)
		{
		case KeyCode::A: return Key::KeyA;
		case KeyCode::B: return Key::KeyB;
		case KeyCode::C: return Key::KeyC;
		case KeyCode::D: return Key::KeyD;
		case KeyCode::E: return Key::KeyE;
		case KeyCode::F: return Key::KeyF;
		case KeyCode::G: return Key::KeyG;
		case KeyCode::H: return Key::KeyH;
		case KeyCode::I: return Key::KeyI;
		case KeyCode::J: return Key::KeyJ;
		case KeyCode::K: return Key::KeyK;
		case KeyCode::L: return Key::KeyL;
		case KeyCode::M: return Key::KeyM;
		case KeyCode::N: return Key::KeyN;
		case KeyCode::O: return Key::KeyO;
		case KeyCode::P: return Key::KeyP;
		case KeyCode::Q: return Key::KeyQ;
		case KeyCode::R: return Key::KeyR;
		case KeyCode::S: return Key::KeyS;
		case KeyCode::T: return Key::KeyT;
		case KeyCode::U: return Key::KeyU;
		case KeyCode::V: return Key::KeyV;
		case KeyCode::W: return Key::KeyW;
		case KeyCode::X: return Key::KeyX;
		case KeyCode::Y: return Key::KeyY;
		case KeyCode::Z: return Key::KeyZ;

		case KeyCode::Num0: return Key::Key0;
		case KeyCode::Num1: return Key::Key1;
		case KeyCode::Num2: return Key::Key2;
		case KeyCode::Num3: return Key::Key3;
		case KeyCode::Num4: return Key::Key4;
		case KeyCode::Num5: return Key::Key5;
		case KeyCode::Num6: return Key::Key6;
		case KeyCode::Num7: return Key::Key7;
		case KeyCode::Num8: return Key::Key8;
		case KeyCode::Num9: return Key::Key9;

		case KeyCode::Keypad0: return Key::NumPad0;
		case KeyCode::Keypad1: return Key::NumPad1;
		case KeyCode::Keypad2: return Key::NumPad2;
		case KeyCode::Keypad3: return Key::NumPad3;
		case KeyCode::Keypad4: return Key::NumPad4;
		case KeyCode::Keypad5: return Key::NumPad5;
		case KeyCode::Keypad6: return Key::NumPad6;
		case KeyCode::Keypad7: return Key::NumPad7;
		case KeyCode::Keypad8: return Key::NumPad8;
		case KeyCode::Keypad9: return Key::NumPad9;

		case KeyCode::F1:  return Key::F1;
		case KeyCode::F2:  return Key::F2;
		case KeyCode::F3:  return Key::F3;
		case KeyCode::F4:  return Key::F4;
		case KeyCode::F5:  return Key::F5;
		case KeyCode::F6:  return Key::F6;
		case KeyCode::F7:  return Key::F7;
		case KeyCode::F8:  return Key::F8;
		case KeyCode::F9:  return Key::F9;
		case KeyCode::F10: return Key::F10;
		case KeyCode::F11: return Key::F11;
		case KeyCode::F12: return Key::F12;

		case KeyCode::LeftArrow:  return Key::Left;
		case KeyCode::RightArrow: return Key::Right;
		case KeyCode::UpArrow:    return Key::Up;
		case KeyCode::DownArrow:  return Key::Down;

		case KeyCode::Home:     return Key::Home;
		case KeyCode::End:      return Key::End;
		case KeyCode::PageUp:   return Key::PageUp;
		case KeyCode::PageDown: return Key::PageDown;

		case KeyCode::Escape:        return Key::Esc;
		case KeyCode::Return:        return Key::Return;
		case KeyCode::KeypadEnter:   return Key::Return;
		case KeyCode::Space:         return Key::Space;
		case KeyCode::Tab:           return Key::Tab;
		case KeyCode::Delete:        return Key::Backspace;
		case KeyCode::ForwardDelete: return Key::Delete;

		case KeyCode::Equals:       return Key::Plus;
		case KeyCode::KeypadPlus:   return Key::Plus;
		case KeyCode::Minus:        return Key::Minus;
		case KeyCode::KeypadMinus:  return Key::Minus;
		case KeyCode::LeftBracket:  return Key::LeftBracket;
		case KeyCode::RightBracket: return Key::RightBracket;
		case KeyCode::Semicolon:    return Key::Semicolon;
		case KeyCode::Quote:        return Key::Quote;
		case KeyCode::Comma:        return Key::Comma;
		case KeyCode::Period:       return Key::Period;
		case KeyCode::Slash:        return Key::Slash;
		case KeyCode::Backslash:    return Key::Backslash;
		case KeyCode::Grave:        return Key::Tilde;

		default: break;
		}

		return Key::None;
	}

	static bool isTextKey(Key::Enum _key)
	{
		switch (_key)
		{
		case Key::None:
		case Key::Esc:
		case Key::Return:
		case Key::Tab:
		case Key::Backspace:
		case Key::Delete:
		case Key::Up:
		case Key::Down:
		case Key::Left:
		case Key::Right:
		case Key::Home:
		case Key::End:
		case Key::PageUp:
		case Key::PageDown:
		case Key::F1:
		case Key::F2:
		case Key::F3:
		case Key::F4:
		case Key::F5:
		case Key::F6:
		case Key::F7:
		case Key::F8:
		case Key::F9:
		case Key::F10:
		case Key::F11:
		case Key::F12:
			return false;

		default:
			return true;
		}
	}

	static uint8_t translateModifiers(const Graphics::ModifierKeys& _modifiers)
	{
		return 0
			| (_modifiers.shift   ? Modifier::LeftShift : 0)
			| (_modifiers.control ? Modifier::LeftCtrl  : 0)
			| (_modifiers.alt     ? Modifier::LeftAlt   : 0)
			| (_modifiers.command ? Modifier::LeftMeta  : 0)
			;
	}

	static MouseButton::Enum translateButton(Graphics::MouseButton _button)
	{
		switch (_button)
		{
		case Graphics::MouseButton::Left:  return MouseButton::Left;
		case Graphics::MouseButton::Right: return MouseButton::Right;
		default:                           return MouseButton::Middle;
		}
	}

	struct Context;
	static Context& getContext();

	/// Off-screen capture, for running the examples without a window taking
	/// the user's focus: EACP_BGFX_CAPTURE=<path.png> runs eacp headless (the
	/// window is made, never shown, never key, no Dock icon), drives every
	/// frame from the main queue into an off-screen Frame over a render
	/// target of the view's own size, scale, depth and stencil,
	/// saves the picture of frame EACP_BGFX_CAPTURE_FRAME (default 120,
	/// counted in frames bgfx rendered) and exits through the normal shutdown.
	/// EACP_BGFX_CAPTURE_HIDPI=1 presses F10 after the first frame, so the
	/// capture is of BGFX_RESET_HIDPI (a pixel-sized backbuffer).
	struct Capture
	{
		Capture()
		{
			if (const char* path = std::getenv("EACP_BGFX_CAPTURE") )
			{
				m_path = path;
			}

			if (const char* frame = std::getenv("EACP_BGFX_CAPTURE_FRAME") )
			{
				m_frame = uint32_t(std::max(1, std::atoi(frame) ) );
			}

			if (const char* hidpi = std::getenv("EACP_BGFX_CAPTURE_HIDPI") )
			{
				m_hidpi = 0 != std::atoi(hidpi);
			}
		}

		bool enabled() const
		{
			return !m_path.empty();
		}

		std::string m_path;
		uint32_t m_frame    = 120;
		uint32_t m_rendered = 0;
		bool     m_saved    = false;
		bool     m_hidpi    = false; // EACP_BGFX_CAPTURE_HIDPI=1: press F10 ("graphics hidpi") once
	};

	struct BgfxView : public GPU::GPUView
	{
		BgfxView()
		{
			setSampleCount(1);

			// bgfx's backbuffer is D24S8: a depth plane for the examples' depth
			// test, and a stencil plane beside it. The renderer gives every
			// window pipeline both, which the pass requires of them.
			setStencil(true);

			setContinuous(true);
			setHandlesMouseEvents(true);
			setGrabsFocusOnMouseDown(true);
		}

		void render(GPU::Frame& _frame) override;
		void resized() override;

		void mouseDown(const Graphics::MouseEvent& _event) override;
		void mouseUp(const Graphics::MouseEvent& _event) override;
		void mouseDragged(const Graphics::MouseEvent& _event) override;
		void mouseMoved(const Graphics::MouseEvent& _event) override;
		void mouseWheel(const Graphics::MouseEvent& _event) override;

		void keyDown(const Graphics::KeyEvent& _event) override;
		void keyUp(const Graphics::KeyEvent& _event) override;
	};

	struct Context
	{
		/// Size events and mouse positions are in points, as entry_osx.mm
		/// reports them, so a Retina display shows the picture stock macOS
		/// does; in pixels once the example asks for BGFX_RESET_HIDPI, which
		/// is then the backbuffer's size. The renderer maps either onto the
		/// drawable (renderer_eacp.cpp's updateWindowScale).
		float eventScale() const
		{
			return bgfx::eacp::isHiDpi() ? m_view->backingScale() : 1.0f;
		}

		void postSize()
		{
			const Graphics::Rect bounds = m_view->getLocalBounds();
			const float scale = eventScale();
			const uint32_t width  = uint32_t(bounds.w*scale + 0.5f);
			const uint32_t height = uint32_t(bounds.h*scale + 0.5f);

			if (0 == width
			||  0 == height
			|| (width == m_width && height == m_height) )
			{
				return;
			}

			m_width  = width;
			m_height = height;
			m_eventQueue.postSizeEvent(kDefaultWindowHandle, width, height);
		}

		void postMouseMove(const Graphics::MouseEvent& _event, bool _locked)
		{
			const float scale = eventScale();

			if (_locked)
			{
				m_eventQueue.postMouseEvent(kDefaultWindowHandle
					, int32_t(_event.delta.x*scale)
					, int32_t(_event.delta.y*scale)
					, m_scroll
					);
				return;
			}

			setMousePos(_event);
			m_eventQueue.postMouseEvent(kDefaultWindowHandle, m_mx, m_my, m_scroll);
		}

		void postMouseButton(const Graphics::MouseEvent& _event, bool _down)
		{
			setMousePos(_event);
			m_eventQueue.postMouseEvent(kDefaultWindowHandle, m_mx, m_my, m_scroll, translateButton(_event.button), _down);
		}

		void postMouseWheel(const Graphics::MouseEvent& _event)
		{
			m_scrollf += _event.preciseScrolling ? _event.delta.y*0.1f : _event.delta.y;
			m_scroll   = int32_t(m_scrollf);
			setMousePos(_event);
			m_eventQueue.postMouseEvent(kDefaultWindowHandle, m_mx, m_my, m_scroll);
		}

		void postKey(const Graphics::KeyEvent& _event, bool _down)
		{
			const Key::Enum key = translateKey(_event.keyCode);

			if (Key::None == key)
			{
				return;
			}

			const uint8_t modifiers = translateModifiers(_event.modifiers);

			if (_down
			&&  isTextKey(key)
			&&  !_event.characters.empty()
			&&  4 >= _event.characters.size() )
			{
				uint8_t chars[4] = {};
				bx::memCopy(chars, _event.characters.data(), _event.characters.size() );
				m_eventQueue.postCharEvent(kDefaultWindowHandle, uint8_t(_event.characters.size() ), chars);
			}

			m_eventQueue.postKeyEvent(kDefaultWindowHandle, key, modifiers, _down);
		}

		void postActivation(bool _active)
		{
			if (_active)
			{
				m_eventQueue.postSuspendEvent(kDefaultWindowHandle, Suspend::WillResume);
				m_eventQueue.postSuspendEvent(kDefaultWindowHandle, Suspend::DidResume);
			}
			else
			{
				m_eventQueue.postSuspendEvent(kDefaultWindowHandle, Suspend::WillSuspend);
				m_eventQueue.postSuspendEvent(kDefaultWindowHandle, Suspend::DidSuspend);
			}
		}

		void requestExit()
		{
			if (!m_exitPosted)
			{
				m_exitPosted = true;
				m_eventQueue.postExitEvent();
			}
		}

		bool isFinished(bgfx::RenderFrame::Enum _result) const
		{
			return bgfx::RenderFrame::NoContext == _result
				&& m_mte.m_done
				;
		}

		void onRendered(bgfx::RenderFrame::Enum _result)
		{
			if (isFinished(_result)
			&&  !m_quitRequested)
			{
				m_quitRequested = true;
				Apps::quit();
			}
		}

		// The API thread's bgfx::shutdown waits on this thread's renderFrame, so
		// once the window is going away (it no longer draws) the frames it waits
		// for are pumped here, with no eacp frame for the renderer to draw into.
		void shutdownAndQuit()
		{
			requestExit();

			while (!isFinished(bgfx::renderFrame(kRenderFrameTimeoutMs) ) )
			{
			}

			if (!m_quitRequested)
			{
				m_quitRequested = true;
				Apps::quit();
			}
		}

		template<typename Func>
		void onMainThread(Func _func)
		{
			Threads::callAsync([this, _func]
				{
					if (NULL != m_window)
					{
						_func(*m_window);
					}
				});
		}

		void setMousePos(const Graphics::MouseEvent& _event)
		{
			const float scale = eventScale();
			const Graphics::Rect bounds = m_view->getLocalBounds();
			m_mx = bx::clamp(int32_t(_event.pos.x*scale), 0, int32_t(bounds.w*scale) );
			m_my = bx::clamp(int32_t(_event.pos.y*scale), 0, int32_t(bounds.h*scale) );
		}

		/// One capture-mode frame, rendered off-screen into a render target
		/// of the view's size, scale, depth and stencil (renderOffscreen). At
		/// the chosen frame the target is read back and written out, after
		/// which the example is asked to exit and frames go on being pumped
		/// until bgfx has shut down.
		///
		/// Not View::renderToImage: it reads the GPU picture as premultiplied
		/// alpha and hands back straight alpha, so wherever bgfx left alpha 0
		/// (13-stencil's clear colour, say) the colour is gone, while the
		/// on-screen layer is opaque and shows it. The read here keeps the
		/// colour and makes the picture opaque, as the screen does.
		void captureTick()
		{
			if (NULL == m_view)
			{
				return;
			}

			if (m_capture.m_hidpi
			&&  1 <= m_capture.m_rendered)
			{
				m_capture.m_hidpi = false;
				m_eventQueue.postKeyEvent(kDefaultWindowHandle, Key::F10, 0, true);
				m_eventQueue.postKeyEvent(kDefaultWindowHandle, Key::F10, 0, false);
			}

			if (!renderOffscreen()
			||  m_capture.m_saved
			||  m_capture.m_rendered < m_capture.m_frame)
			{
				return;
			}

			m_capture.m_saved = true;

			const int width  = m_captureTarget->width();
			const int height = m_captureTarget->height();

			Graphics::ImageData pixels;
			pixels.resize(size_t(width)*size_t(height)*4);
			m_captureTarget->read(pixels.data() );

			for (size_t ii = 0; ii + 3 < pixels.size(); ii += 4)
			{
				std::swap(pixels[ii+0], pixels[ii+2]); // BGRA to RGBA
				pixels[ii+3] = 255;
			}

			Graphics::Image(width, height, std::move(pixels) ).save(m_capture.m_path);
			BX_TRACE("eacp capture: frame %d, %dx%d, saved to %s", m_capture.m_rendered, width, height, m_capture.m_path.c_str() );

			requestExit();
		}

		bool renderOffscreen()
		{
			const Graphics::Rect bounds = m_view->getLocalBounds();
			const float scale = m_view->backingScale();
			const int width  = int(bounds.w*scale + 0.5f);
			const int height = int(bounds.h*scale + 0.5f);

			if (0 >= width
			||  0 >= height)
			{
				return false;
			}

			if (!m_captureTarget
			||  m_captureTarget->width()  != width
			||  m_captureTarget->height() != height)
			{
				GPU::TextureDescriptor desc;
				desc.width        = width;
				desc.height       = height;
				desc.format       = GPU::TextureFormat::BGRA8Unorm;
				desc.renderTarget = true;
				desc.depth        = m_view->hasDepth();
				desc.stencil      = m_view->hasStencil();
				m_captureTarget.emplace(GPU::Device::shared().makeTexture(desc, NULL) );
			}

			GPU::OffscreenTarget target;
			target.colorTexture = m_captureTarget->nativeTexture();
			target.depthTexture = m_captureTarget->nativeDepthTexture();

			{
				// Committed and waited for when it goes out of scope, so the
				// target is readable after.
				GPU::Frame frame(GPU::Device::shared(), target, scale);
				m_view->render(frame);
			}

			return true;
		}

		EventQueue m_eventQueue;
		Capture m_capture;
		std::optional<GPU::Texture> m_captureTarget;
		MainThreadEntry m_mte;
		BgfxView* m_view = NULL;
		Graphics::Window* m_window = NULL;

		uint32_t m_width  = 0;
		uint32_t m_height = 0;
		float   m_scrollf = 0.0f;
		int32_t m_mx      = 0;
		int32_t m_my      = 0;
		int32_t m_scroll  = 0;
		bool m_exitPosted    = false;
		bool m_quitRequested = false;
	};

	static Context s_ctx;

	static Context& getContext()
	{
		return s_ctx;
	}

	static Graphics::WindowOptions makeWindowOptions()
	{
		Graphics::WindowOptions options;
		options.title  = "bgfx";
		options.width  = ENTRY_DEFAULT_WIDTH;
		options.height = ENTRY_DEFAULT_HEIGHT;
		options.flags  =
		{
			Graphics::WindowFlags::Titled,
			Graphics::WindowFlags::Closable,
			Graphics::WindowFlags::Miniaturizable,
			Graphics::WindowFlags::Resizable,
		};
		options.backgroundColor = Graphics::Color::black();
		options.onQuit = [] { getContext().shutdownAndQuit(); };

		if (getContext().m_capture.enabled() )
		{
			// Never key, never in the Dock or the app switcher; hidden as soon
			// as it exists (App's constructor).
			Apps::setDockIconVisible(false);
			options.showInactive = true;
		}

		return options;
	}

	struct App
	{
		App()
			: m_window(m_view, makeWindowOptions() )
		{
			s_ctx.m_view   = &m_view;
			s_ctx.m_window = &m_window;

			m_window.events.onActivationChanged = [](bool _active) { getContext().postActivation(_active); };

			Apps::setQuitHandler([]
				{
					getContext().shutdownAndQuit();
					return false;
				});

			if (s_ctx.m_capture.enabled() )
			{
				// The display link stays out of it; every frame is an
				// off-screen frame from the capture chain (sample count 1, as
				// the view is).
				m_view.setContinuous(false);
				m_window.setVisible(false);
			}
			else
			{
				m_view.focus();
			}

			// Marks this thread as bgfx's render thread before the API thread
			// can call bgfx::init, so bgfx creates no render thread of its own.
			bgfx::renderFrame();

			s_ctx.postSize();

			m_thread.init(MainThreadEntry::threadFunc, &s_ctx.m_mte, 0, "Entry Thread");

			if (s_ctx.m_capture.enabled() )
			{
				// A chain of main-queue calls rather than a timer: a hidden,
				// inactive app is App Napped, and its timers stop firing.
				scheduleCapture();
			}
		}

		static void scheduleCapture()
		{
			Threads::callAsync([]
				{
					if (!getContext().m_quitRequested)
					{
						getContext().captureTick();
						scheduleCapture();
					}
				});
		}

		~App()
		{

			if (!s_ctx.m_mte.m_done)
			{
				s_ctx.shutdownAndQuit();
			}

			m_thread.shutdown();
			s_ctx.m_window = NULL;
			s_ctx.m_view   = NULL;
		}

		BgfxView m_view;
		Graphics::Window m_window;
		bx::Thread m_thread;
	};

	void BgfxView::render(GPU::Frame& _frame)
	{
		const bgfx::RenderFrame::Enum result = bgfx::eacp::renderFrame(_frame, kRenderFrameTimeoutMs);

		if (bgfx::RenderFrame::Render == result)
		{
			++getContext().m_capture.m_rendered;

			// A BGFX_RESET_HIDPI change switches the size events between
			// points and pixels.
			getContext().postSize();
		}

		getContext().onRendered(result);
	}

	void BgfxView::resized()
	{
		getContext().postSize();
	}

	void BgfxView::mouseDown(const Graphics::MouseEvent& _event)
	{
		getContext().postMouseButton(_event, true);
	}

	void BgfxView::mouseUp(const Graphics::MouseEvent& _event)
	{
		getContext().postMouseButton(_event, false);
	}

	void BgfxView::mouseDragged(const Graphics::MouseEvent& _event)
	{
		mouseMoved(_event);
	}

	void BgfxView::mouseMoved(const Graphics::MouseEvent& _event)
	{
		const Graphics::Window* window = getWindow();
		getContext().postMouseMove(_event, NULL != window && window->isMouseLocked() );
	}

	void BgfxView::mouseWheel(const Graphics::MouseEvent& _event)
	{
		getContext().postMouseWheel(_event);
	}

	void BgfxView::keyDown(const Graphics::KeyEvent& _event)
	{
		getContext().postKey(_event, true);
	}

	void BgfxView::keyUp(const Graphics::KeyEvent& _event)
	{
		getContext().postKey(_event, false);
	}

	const Event* poll()
	{
		return s_ctx.m_eventQueue.poll();
	}

	const Event* poll(WindowHandle _handle)
	{
		return s_ctx.m_eventQueue.poll(_handle);
	}

	void release(const Event* _event)
	{
		s_ctx.m_eventQueue.release(_event);
	}

	WindowHandle createWindow(int32_t _x, int32_t _y, uint32_t _width, uint32_t _height, uint32_t _flags, const char* _title)
	{
		BX_UNUSED(_x, _y, _width, _height, _flags, _title);
		WindowHandle handle = { UINT16_MAX };
		return handle;
	}

	void destroyWindow(WindowHandle _handle)
	{
		BX_UNUSED(_handle);
	}

	void setWindowPos(WindowHandle _handle, int32_t _x, int32_t _y)
	{
		if (kDefaultWindowHandle.idx == _handle.idx)
		{
			s_ctx.onMainThread([=](Graphics::Window& _window) { _window.setPosition({ float(_x), float(_y) }); });
		}
	}

	// In points, as entry_osx.mm takes it, and as size events report it
	// unless the example asked for BGFX_RESET_HIDPI.
	void setWindowSize(WindowHandle _handle, uint32_t _width, uint32_t _height)
	{
		if (kDefaultWindowHandle.idx == _handle.idx)
		{
			s_ctx.onMainThread([=](Graphics::Window& _window) { _window.setSize({ float(_width), float(_height) }); });
		}
	}

	void setWindowTitle(WindowHandle _handle, const char* _title)
	{
		if (kDefaultWindowHandle.idx == _handle.idx)
		{
			const std::string title = _title;
			s_ctx.onMainThread([=](Graphics::Window& _window) { _window.setTitle(title); });
		}
	}

	void setWindowFlags(WindowHandle _handle, uint32_t _flags, bool _enabled)
	{
		BX_UNUSED(_handle, _flags, _enabled);
	}

	// eacp's Window has no runtime fullscreen toggle on any platform yet.
	void toggleFullscreen(WindowHandle _handle)
	{
		BX_UNUSED(_handle);
	}

	void setMouseLock(WindowHandle _handle, bool _lock)
	{
		if (kDefaultWindowHandle.idx == _handle.idx)
		{
			s_ctx.onMainThread([=](Graphics::Window& _window) { _window.setMouseLocked(_lock); });
		}
	}

	void* getNativeWindowHandle(WindowHandle _handle)
	{
		if (kDefaultWindowHandle.idx == _handle.idx)
		{
			return static_cast<GPU::GPUView*>(s_ctx.m_view);
		}

		return NULL;
	}

	void* getNativeDisplayHandle()
	{
		return NULL;
	}

	bgfx::NativeWindowHandleType::Enum getNativeWindowHandleType()
	{
		return bgfx::NativeWindowHandleType::Default;
	}

	// TODO: gamepads, from eacp's GameInput snapshots into GamepadEvents
	// (plan.md 4.9).

} // namespace entry

int main(int _argc, const char* const* _argv)
{
	entry::s_ctx.m_mte.m_argc = _argc;
	entry::s_ctx.m_mte.m_argv = _argv;

	if (entry::s_ctx.m_capture.enabled() )
	{
		// Windows are made but never shown (eacp's headless mode, the one its
		// test binaries run in).
		::eacp::Apps::setHeadless(true);
	}

	return ::eacp::Apps::run<entry::App>();
}

#endif // ENTRY_CONFIG_USE_EACP
