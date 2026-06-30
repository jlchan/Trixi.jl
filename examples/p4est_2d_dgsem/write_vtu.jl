using Trixi2Vtk
# Paths
const BASE_DIR = "/Users/jchan/.julia/dev/Trixi/out_from_sunny/mach7_scaling2_refinement1"
const INPUT_DIR = joinpath(BASE_DIR, ".")
const OUTPUT_DIR = joinpath(BASE_DIR, "vtu")
# Trixi2Vtk looks for mesh.h5 next to each solution file
mesh_file = joinpath(INPUT_DIR, "mesh.h5")
isfile(mesh_file) || error("Missing $mesh_file — copy it from the original run's out/ directory")
mkpath(OUTPUT_DIR)

# Convert all snapshots; also writes solution.pvd (+ solution_celldata.pvd)
trixi2vtk(
    joinpath(INPUT_DIR, "solution_*.h5");
    output_directory = OUTPUT_DIR,
    nvisnodes = 12,          # polydeg was 3; 12 gives smoother ParaView images
    pvd = "solution",
    verbose = true,
)
println("Done. Open ", joinpath(OUTPUT_DIR, "solution.pvd"), " in ParaView.")




const PVPYTHON = "/Applications/ParaView-6.0.1.app/Contents/bin/pvpython"
const MAKE_MOVIE = joinpath(@__DIR__, "make_movie.py")

println("Running make_movie.py...")
if !success(run(`$PVPYTHON $MAKE_MOVIE`))
    error("make_movie.py failed")
end
println("Movie frames written.")