/*
 * Copyright 2011-2026 Branimir Karadzic. All rights reserved.
 * License: https://github.com/bkaradzic/bgfx/blob/master/LICENSE
 */

#ifndef BGFX_EACP_H_HEADER_GUARD
#define BGFX_EACP_H_HEADER_GUARD

#include "bgfx.h"

namespace eacp { namespace GPU { class Frame; } }

namespace bgfx { namespace eacp
{
	/// Render frame into an eacp frame.
	///
	/// The eacp renderer draws into whichever `eacp::GPU::Frame` is current
	/// while `bgfx::renderFrame` runs, and holds none across calls. Call this
	/// from `eacp::GPU::GPUView::render`, on eacp's main thread, in place of
	/// `bgfx::renderFrame`: it makes `_frame` current, renders, and clears it
	/// again before returning, so the frame presents when `render` returns.
	///
	/// @param[in] _frame The frame `GPUView::render` was handed.
	/// @param[in] _msecs Timeout in milliseconds, as for `bgfx::renderFrame`.
	///
	/// @returns Current renderer context state. See: `bgfx::RenderFrame`.
	///
	/// @attention Only available with `BGFX_CONFIG_RENDERER_EACP`.
	///
	RenderFrame::Enum renderFrame(::eacp::GPU::Frame& _frame, int32_t _msecs = -1);

	/// Whether the main swap chain asks for `BGFX_RESET_HIDPI`, as of the
	/// last frame rendered.
	///
	/// bgfx's backbuffer is the size `bgfx::reset` is given, and the eacp
	/// renderer maps it onto the view's drawable, which is in pixels. An
	/// entry layer reports the window size, and mouse positions, in points
	/// when this is false (the picture is the one stock macOS shows, drawn
	/// at the drawable's resolution) and in pixels when it is true (a pixel
	/// sized backbuffer). Read it on eacp's main thread after `renderFrame`.
	///
	/// @attention Only available with `BGFX_CONFIG_RENDERER_EACP`.
	///
	bool isHiDpi();

} /* namespace eacp */ } // namespace bgfx

#endif // BGFX_EACP_H_HEADER_GUARD
