module my_Kitaev

using Yao
# using YaoBlocks
#using YaoExtensions
#using GalacticFlux
using Optimisers
using LinearAlgebra
using SparseArrays
using Random
using PyCall
using DelimitedFiles
using PyPlot
using KrylovKit
lalg = pyimport("scipy.sparse.linalg")



const GPU_enabled = "no"

if GPU_enabled == "yes"
    using CuYao
elseif GPU_enabled == "no"
    using Yao
else
    @warn "GPU/CPU flag must be yes or no string"
end

include("kitaev_h_and_funcs.jl")
include("state_prep.jl")
include("vqe_definitions.jl")
include("QFD_funcs.jl")
include("GF_funcs.jl")

export gridproperties, edges, kitaev, getKij, getstabilizers, getlogicstrings, getLijk, getplaquetteparityops # lattice and operator functions
export magnX, magnY, magnZ, prtZ, prtY, prtX # observable operators
export get_spectrum, printkrylovresults, printinit, printvqeresults, vecexpect # printing functions
export rsplit2, zero_small!, chopcomplex # auxiliary functions

export prepare_inital_state
export measure_selected!, stabilize_plaquettes!, stabilize_logic!, excite_vortices!, stabilize_all_plaquettes!

export vqe_centralizer_ansatz, trueground, prepare_for_vqe, perform_vqe

export get_kappa, find_GS_QFD, find_GS_trot, output_ham_properties, trot_circ

export creation_op, GF_coeffs, Greens_fn, GS_imag_evol, check_GF, exact_GF, QFD_GF, DSF_exact, DSF_QFD, compare_GFS, Greens_fn_trot, DSF_QFD_trot, DSF_exact_nn, DSF_QFD_nn, DSF_exact_nn_range_freqs, DSF_QFD_nn_range_freqs, DSF_QFD_nn_range_freqs_quick

end
