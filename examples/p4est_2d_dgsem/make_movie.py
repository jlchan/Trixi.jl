from paraview.simple import *

base = "/Users/jchan/.julia/dev/Trixi/out_from_sunny/mach7_scaling2_refinement1"
pvd_rho = f"{base}/vtu/solution.pvd"
pvd_celldata = f"{base}/vtu/solution_celldata.pvd"
output_prefix = f"{base}/mach7_rho_indicator"

reader_rho = OpenDataFile(pvd_rho)
reader_ind = OpenDataFile(pvd_celldata)
reader_rho.UpdatePipeline()
reader_ind.UpdatePipeline()

RemoveViewsAndLayouts()
layout = CreateLayout("Stacked")
layout.SetSize(3480, 1920)

view_rho = CreateView("RenderView")
layout.AssignView(0, view_rho)
top_loc = layout.SplitViewHorizontal(view=view_rho, fraction=0.5)
# top_loc = layout.SplitViewVertical(view=view_rho, fraction=0.5)
view_ind = CreateView("RenderView")
layout.AssignView(top_loc + 1, view_ind)
AddCameraLink(view_rho, view_ind)

for view in (view_rho, view_ind):
    view.OrientationAxesVisibility = 0
    view.UseLight = 0
    view.CameraParallelProjection = 1

display_rho = Show(reader_rho, view_rho)
ColorBy(display_rho, ("POINTS", "rho"))
GetColorTransferFunction("rho").RescaleTransferFunction(0.0, 7.0)

display_ind = Show(reader_ind, view_ind)
ColorBy(display_ind, ("CELLS", "indicator_shock_capturing"))
GetColorTransferFunction("indicator_shock_capturing").RescaleTransferFunction(0.0, 1.0)
display_ind.SetScalarBarVisibility(view_ind, True)
ind_lut = GetColorTransferFunction("indicator_shock_capturing")
ind_bar = GetScalarBar(ind_lut, view_ind)

ind_bar.AutoOrient = 0
ind_bar.Orientation = "Horizontal"
ind_bar.Title = ""
ind_bar.ComponentTitle = ""

# optional: place horizontal bar along bottom of indicator panel
ind_bar.WindowLocation = "Any Location"
ind_bar.Position = [0.25, 0.05]

timesteps = reader_rho.TimestepValues

# After Show(...) and before the loop:
view_rho.ViewTime = timesteps[0]
view_ind.ViewTime = timesteps[0]
reader_rho.UpdatePipeline()
reader_ind.UpdatePipeline()

for i, t in enumerate(timesteps):
    print(f"Timestep {i + 1}/{len(timesteps)} (t={t})", flush=True)
    view_rho.ViewTime = t
    view_ind.ViewTime = t

    # reader_ind.UpdatePipeline(time=t)   # important: flush pipeline at this t
    # display_ind.RescaleTransferFunctionToDataRange(False, True)

    if i == 0:
        view_rho.ResetCamera(True, 1.8)   # linked camera updates both views
        view_rho.ResetCamera(True, 1.8)   # linked camera updates both views
    RenderAllViews()
    SaveScreenshot(f"{output_prefix}.{i:04d}.png", layout, ImageResolution=[3480, 1920])


# Run from the Trixi repo root:
#   /Applications/ParaView-6.0.1.app/Contents/bin/pvpython examples/p4est_2d_dgsem/make_movie.py
#
# Then stitch the PNG sequence into a movie (requires ffmpeg):
#   cd out_from_sunny/mach3_scaling2_coarse
#   ffmpeg -framerate 12 -i mach3_rho_indicator.%04d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p mach3_rho_indicator.mp4
