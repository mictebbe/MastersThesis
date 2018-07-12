import cairo
import librsvg


width = 256
height = 256

svg = rsvg.Handle('cool.svg')
unscaled_width = svg.props.width
unscaled_height = svg.props.height

svg_surface = cairo.SVGSurface(None, width, height)
svg_context = cairo.Context(svg_surface)
svg_context.save()
svg_context.scale(width/unscaled_width, height/unscaled_height)
svg.render_cairo(svg_context)
svg_context.restore()

svg_surface.write_to_png('cool.png')