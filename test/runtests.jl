using DeepDriveWE
using Test

@testset "DeepDriveWE.jl" begin
    include("test_api.jl")
    include("test_binners.jl")
    include("test_recyclers.jl")
    include("test_resamplers.jl")
    include("test_alanine_dipeptide.jl")
    include("test_idp_fragment.jl")
    include("test_driver.jl")
end
