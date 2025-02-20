# function to calculate honeycomb lattice properties and use it for simulation
function gridproperties(Lx::Int, Lz::Int, BC)
    if (BC < 0) || (BC > 1)
        @warn "Boundary condition flag is ill-defined (shall be 0 or 1)"
    end
    if (Lx <= 0) || (Lz <= 0)
        @warn "Number of plaquettes Lx/Lz is ill-defined (shall be greater than zero)"
    end
    Nx = 4 + 2 * (Lx - 1) # number of vertices in horizontal x direction # added auxiliary vertices to the main lattice
    Nz = 2 + (Lz - 1) # number of vertices in vertical z direction
    Nv = Lz * (4 * Lx + 2) - 2 * Lx * (Lz - 1) + 2 * BC # total number of vertices for rectangular hex grid
    return Nv, Nx, Nz
end

# function to form a hexagonal lattice (with open or toric boundary) and return bonds to formulate the Kitaev Hamiltonian
function edges(Lx::Int, Lz::Int, BC) # only periodic BC is considered
    if (BC < 0) || (BC > 1)
        @warn "Boundary condition flag is ill-defined (shall be 0 or 1)"
    end
    if (Lx <= 0) || (Lz <= 0)
        @warn "Number of plaquettes Lx/Lz is ill-defined (shall be greated than zero)"
    end
    Nx = 2 * (Lx + 1) # number of vertices in the x direction (horizontal)
    Nz = (Lz + 1) # number of vertices in the z direction (vertical)
    Nv = Lz * (4 * Lx + 2) - 2 * Lx * (Lz - 1) + 2 * BC # total number of vertices for rectangular hex grid
    # introduce main XX edges
    XXvec = Vector{Int64}()
    for j = 1:(Lx+1)
        for k = 1:(Lz+1)
            append!(XXvec, [(2*j +(k-1)*Nx)  (mod(2*j+1, 1:Nx) + (k-1)*Nx)])
        end
    end
    # introduce main YY edges
    YYvec = Vector{Int64}()
    for j = 1:2:(2*(Lx+1)*(Lz+1))
        append!(YYvec, [(j)  (j+1)])
    end
    # introduce main ZZ edges
    ZZvec = Vector{Int64}()
    for j = 1:(Lx+1)
        for k = 1:Nz
        append!(ZZvec, [(2*j-1 +(k-1)*Nx)  (mod(2*j-2, 1:Nx) + mod(k, 0:Lz)*Nx)])
        end
    end
    # split vectors of vertices into separate bonds of three flavours
    XXedges = rsplit2(XXvec, 2)
    YYedges = rsplit2(YYvec, 2)
    ZZedges = rsplit2(ZZvec, 2)
    return XXedges, YYedges, ZZedges
end

# define Kitaev Hamiltonian on the honeycomb lattice with generic JX, JY, JZ
# n is a number of qubits (vertices), and also use specfied bonds
function kitaev(n::Int, XXbonds::Vector{Array{Int64,1}}, YYbonds::Vector{Array{Int64,1}}, ZZbonds::Vector{Array{Int64,1}}, J::Vector{Float64}, hfield::Vector{Float64})
    HX = sum([J[1] * put(n, XXbonds[i][1] => X) * put(n, XXbonds[i][2] => X) for i = 1:length(XXbonds)])
    HY = sum([J[2] * put(n, YYbonds[i][1] => Y) * put(n, YYbonds[i][2] => Y) for i = 1:length(YYbonds)])
    HZ = sum([J[3] * put(n, ZZbonds[i][1] => Z) * put(n, ZZbonds[i][2] => Z) for i = 1:length(ZZbonds)])
    HXfield = sum([hfield[1] * put(n, i => X) for i = 1:n])
    HYfield = sum([hfield[2] * put(n, i => Y) for i = 1:n])
    HZfield = sum([hfield[3] * put(n, i => Z) for i = 1:n])
    return HX + HY + HZ + HXfield + HYfield + HZfield
end

# define plaquette stabilizer operators w = [list of len(6) vectors], corresponding index sequences, and number of vortices operator
function getstabilizers(Lx::Int, Lz::Int, BC, n)
    Nx = 2 * (Lx + 1) # number of vertices in the x direction (horizontal)
    Nz = (Lz + 1) # number of vertices in the z direction (vertical)
    Nv = Lz * (4 * Lx + 2) - 2 * Lx * (Lz - 1) + 2 * BC # total number of vertices for rectangular hex grid
    w = Array{ChainBlock, 1}(undef, (Lx+1)*(Lz+1))
    plaq_seq = Array{Vector, 1}(undef, (Lx+1)*(Lz+1))
    if BC == 1
        for j = 1:(Lx+1)
            for k = 1:(Lz+1)
                plaq_seq[j + (k-1)*(Lx+1)] = [(2*j-1+(k-1)*Nx), (2*j+(k-1)*Nx), (mod(2*j+1, 1:Nx)+(k-1)*Nx), (mod(2*j, 1:Nx)+(mod(k, 0:(Nz-1)))*Nx) , (mod(2*j-1, 1:Nx)+(mod(k, 0:(Nz-1)))*Nx), (mod(2*j-2, 1:Nx)+(mod(k, 0:(Nz-1)))*Nx)];
                w[j + (k-1)*(Lx+1)] = put(Nv, (2*j-1+(k-1)*Nx) => X) * put(Nv, (2*j+(k-1)*Nx) => Z) * put(Nv, (mod(2*j+1, 1:Nx)+(k-1)*Nx) => Y) * put(Nv, (mod(2*j, 1:Nx)+(mod(k, 0:(Nz-1)))*Nx) => X) *
                put(Nv, (mod(2*j-1, 1:Nx)+(mod(k, 0:(Nz-1)))*Nx) => Z) * put(Nv, (mod(2*j-2, 1:Nx)+(mod(k, 0:(Nz-1)))*Nx) => Y);
            end
        end
        vtot = (n*igate(Nv) - 2*sum(w))/4
    else
        @warn "Plaquettes so far only work now for BC == 1"
    end
    return w, plaq_seq, vtot
end

function geteffectivefield(n, plaq_seq, K)
    heff = 0.0*put(n, 1=>I2)
    for pl in plaq_seq
        heff += put(n, pl[1]=>Y)*put(n, pl[2]=>Z)*put(n, pl[3]=>X)
        heff += put(n, pl[2]=>X)*put(n, pl[3]=>Y)*put(n, pl[4]=>Z)
        heff += put(n, pl[3]=>Z)*put(n, pl[4]=>X)*put(n, pl[5]=>Y)
        heff += put(n, pl[4]=>Y)*put(n, pl[5]=>Z)*put(n, pl[6]=>X)
        heff += put(n, pl[5]=>X)*put(n, pl[6]=>Y)*put(n, pl[1]=>Z)
        heff += put(n, pl[6]=>Z)*put(n, pl[1]=>X)*put(n, pl[2]=>Y)
    end
    return K*heff
end

# find indices for logic string in Lx and Lz direction
function getlogicstrings(plaq_seq,Lx,Lz)
    LXstr = Vector{Int}()
    for lx = 1:Lx+1
        push!(LXstr, plaq_seq[lx][6])
        push!(LXstr, plaq_seq[lx][5])
        push!(LXstr, plaq_seq[lx][4])
    end
    LZstr = Vector{Int}()
    for lz = 1:Lz+1
        push!(LZstr, plaq_seq[(lz-1)*(Lx+1) + 1][2])
        push!(LZstr, plaq_seq[(lz-1)*(Lx+1) + 1][3])
        push!(LZstr, plaq_seq[(lz-1)*(Lx+1) + 1][4])
    end
    return unique(LXstr), unique(LZstr)
end

# write the operators at each bond that form a centralizer group of Pauli string for stabilizers in S
function getKij(n::Int, XXbonds::Vector{Array{Int64,1}}, YYbonds::Vector{Array{Int64,1}}, ZZbonds::Vector{Array{Int64,1}})
    KXij = [put(n, XXbonds[i][1] => X) * put(n, XXbonds[i][2] => X) for i = 1:length(XXbonds)]
    KYij = [put(n, YYbonds[i][1] => Y) * put(n, YYbonds[i][2] => Y) for i = 1:length(YYbonds)]
    KZij = [put(n, ZZbonds[i][1] => Z) * put(n, ZZbonds[i][2] => Z) for i = 1:length(ZZbonds)]
    # output all centralizers at XX, YY, ZZ bonds as a single list
    return collect(Iterators.flatten([KXij, KYij, KZij]))
end

# prepare a list of 2 centralizer products (3-site centralizers) for each plaquette
function getLijk(Lx::Int, Lz::Int, BC)
    Nx = 2 * (Lx + 1) # number of vertices in the x direction (horizontal)
    Nz = (Lz + 1) # number of vertices in the z direction (vertical)
    Nv = Lz * (4 * Lx + 2) - 2 * Lx * (Lz - 1) + 2 * BC # total number of vertices for rectangular hex grid
    plaq_seq = Array{Vector, 1}(undef, (Lx+1)*(Lz+1))
    if BC == 1
        for j = 1:(Lx+1)
            for k = 1:(Lz+1)
                plaq_seq[j + (k-1)*(Lx+1)] = [(2*j-1+(k-1)*Nx), (2*j+(k-1)*Nx), (mod(2*j+1, 1:Nx)+(k-1)*Nx), (mod(2*j, 1:Nx)+(mod(k, 0:(Nz-1)))*Nx) , (mod(2*j-1, 1:Nx)+(mod(k, 0:(Nz-1)))*Nx), (mod(2*j-2, 1:Nx)+(mod(k, 0:(Nz-1)))*Nx)]
            end
        end
    else
        @warn "Plaquettes so far only work now for BC == 1"
    end
    Lijk = Array{ChainBlock, 1}(undef, 6*length(plaq_seq))
    Lijk_indices = Array{Vector, 1}(undef, 6*length(plaq_seq))
    plqops = [[Y, Z, X], [X, Y, Z], [Z, X, Y], [Y, Z, X], [X, Y, Z], [Z, X, Y]]
    for k = 1:length(plaq_seq)
        for i = 1:6
            Lijk[6*(k-1)+i] = chain(Nv, put(plaq_seq[k][(i-1)%6+1]=>plqops[(i-1)%6+1][1]), put(plaq_seq[k][(i-1+1)%6+1]=>plqops[(i-1)%6+1][2]), put(plaq_seq[k][(i-1+2)%6+1]=>plqops[(i-1)%6+1][3]))
            Lijk_indices[6*(k-1)+i] = [plaq_seq[k][(i-1)%6+1], plaq_seq[k][(i-1+1)%6+1], plaq_seq[k][(i-1+2)%6+1]]
        end
    end
    return Lijk, Lijk_indices
end


function getLXstr(LXverts)
    LXstr = chain(n)
    for i = 1:length(LXverts)
        push!(LXstr, put(LXverts[i]=>Z) )
    end
    return LXstr
end

function getLZstr(LZverts)
    LZstr = chain(n)
    for i = 1:length(LZverts)
        push!(LZstr, put(LZverts[i]=>Y) )
    end
    return LZstr
end

# get parity operators as {Z, Y, X} len(6) strings at each plaquette k, and combine in 3 lists
function getplaquetteparityops(plaquette_sequences)
    plqtprtZ = Array{ChainBlock, 1}(undef, (Lx+1)*(Lz+1))
    plqtprtY = Array{ChainBlock, 1}(undef, (Lx+1)*(Lz+1))
    plqtprtX = Array{ChainBlock, 1}(undef, (Lx+1)*(Lz+1))
    for k = 1:length(plaquette_sequences)
        plqtprtZ[k] = prod([put(n, i => Z) for i in plaquette_sequences[k]])
        plqtprtY[k] = prod([put(n, i => Y) for i in plaquette_sequences[k]])
        plqtprtX[k] = prod([put(n, i => X) for i in plaquette_sequences[k]])
    end
    return plqtprtZ, plqtprtY, plqtprtX
end

# define total magnetization operators
magnZ(n::Int) = sum([put(n, i => Z) for i = 1:n]) # define total magnetization (unnormalized) in Z direction
magnY(n::Int) = sum([put(n, i => Y) for i = 1:n]) # define total magnetization (unnormalized) in Y direction
magnX(n::Int) = sum([put(n, i => X) for i = 1:n]) # define total magnetization (unnormalized) in X direction

# define total parity operators
prtZ(n::Int) = prod([put(n, i => Z) for i = 1:n]) # define parity operator as Z Pauli string for all sites
prtY(n::Int) = prod([put(n, i => Y) for i = 1:n]) # define parity operator as Y Pauli string for all sites
prtX(n::Int) = prod([put(n, i => X) for i = 1:n]) # define parity operator as X Pauli string for all sites

# get expectation value of operator for a vector state
vecexpect(operator, s, eps::Float64) = real(chopcomplex(s' * mat(operator) * s, eps))

# prepare a print function to show properties of low-energy states coming from Krylov diagonalization
function printkrylovresults(eigvectors, eigenergies)
    eps = 1e-12; # used for truncation of complex numbers
    for i = 1:length(eigvectors)
        s = eigvectors[i]
        #s = sparse(vectors[i])
        #SparseArrays.dropzeros!(s)
        println("i = $i eigenstate has energy E = $(real(eigenergies[i]))")
        println("Krylov magnetization is Mz = $(vecexpect(magnZ(n), s, eps)), My = $(vecexpect(magnY(n), s, eps)), Mx = $(vecexpect(magnX(n), s, eps))")
        println("Total parities are Pz = $(vecexpect(prtZ(n), s, eps)), Py = $(vecexpect(prtY(n), s, eps)), Px = $(vecexpect(prtX(n), s, eps)) with sum Ptot = $(vecexpect(prtZ(n)+prtY(n)+prtX(n), s, eps))")
        if BC == 1
            wstring = join(["⟨w$k⟩ = $(vecexpect(w[k], s, eps)); " for k=1:length(w)])
            println("Total number of vortices is ⟨V⟩ = $(vecexpect(vtot, s, eps)) ")
        else
            println("Yet only BC==1 was considered")
        end
        println(" ")
    end
end

# diagonalize the low-energy part of the Hamiltonian using Krylov method
# https://jutho.github.io/KrylovKit.jl/stable/man/eig/#KrylovKit.eigsolve
function get_spectrum(hamiltonian, ifprint::Bool)
    # need to be careful -- sometimes Krylov does not converge for all states that are degenerate -> increase tol and krylovdim
    # println("Calculating eigenspectrum with the Krylov approach...")
    energies, vectors, info = eigsolve(mat(hamiltonian), 5, :SR, ishermitian = true, tol = 1e-12, verbosity = 1, krylovdim = 2^7)#, verbosity = 2, tol = 1e-12, krylovdim = 2^7)
    # println("Smallest eigenvalues of the Kitaev Hamiltonian are ", energies)
    if ifprint
        # print expectations for low-energy states from Krylov [takes a lot of time for large system]
        printkrylovresults(vectors, energies)
    end
    return energies, vectors
end

# function for printing properties of initial stabilizer state
function printinit(state)
    wstring = join(["⟨w$k⟩ = $(real(expect(w[k], state))); " for k=1:length(w)])
    println("Stabilizer state energy = $(real(expect(h, state))) \r
Stabilizer state magnetization: [ Mz = $(real(expect(magnZ(n), state))) | My = $(real(expect(magnY(n), state))) | Mx = $(real(expect(magnX(n), state))) ] \r
Stabilizer state parities: [ Pz = $(real(expect(prtZ(n), state))) | Py = $(real(expect(prtY(n), state))) | Px = $(real(expect(prtX(n), state))) ] \r
Stabilizer state summed parity : ⟨Ptot⟩ = $(real(expect(prtZ(n)+prtY(n)+prtX(n), state))) \r
Stabilizer state total number of vortices: ⟨V⟩ = $(real(expect(vtot, state))) and individual plaquettes are \r
$wstring \r
    ")
end

# function for printing final VQE state properties and energies coming from Krylov diagonalization
function printvqeresults(state, energies)
    wstring = join(["⟨w$k⟩ = $(real(expect(w[k], state))); " for k=1:length(w)])
    println("Final VQE energy = $(real(expect(h, state))) \r
Final VQE magnetization: [ Mz = $(real(expect(magnZ(n), state))) | My = $(real(expect(magnY(n), state))) | Mx = $(real(expect(magnX(n), state))) ] \r
Final VQE parities: [ Pz = $(real(expect(prtZ(n), state))) | Py = $(real(expect(prtY(n), state))) | Px = $(real(expect(prtX(n), state))) ] \r
Final VQE summed parity : ⟨Ptot⟩ = $(real(expect(prtZ(n)+prtY(n)+prtX(n), state))) \r
Final VQE total number of vortices: ⟨V⟩ = $(real(expect(vtot, state))) and individual plaquettes are \r
$wstring  \r
Smallest eigenvalues of the Kitaev Hamiltonian are $energies \r
    ")
end


# function to split a vector of vertices into pairs of vertices on the same bond
function rsplit2(v, l::Int)
    m = reshape(v, l, div(length(v), l))
    return [m[:, i] for i = 1:size(m, 2)]
end

trial= [2,3,4,5,6,7]
trial2 = reshape(trial,2,3)
trial2[:,1]
rsplit2(trial,2)
# function to chop dense vector up to given tolerance
function zero_small!(M, tol::Float64)
    for ι in eachindex(M)
        if abs(M[ι]) ≤ tol
            M[ι] = 0
        end
    end
    M
end

# chop to zero for small abs, or chop re/im parts
function chopcomplex(c::Complex, tol::Float64)
    if abs(c) ≤ tol
        c = 0.0
    end
    if (real(c) ≤ tol) && (imag(c) ≥ tol)
        c = round(imag(c), digits = 3)
    end
    if (real(c) ≥ tol) && (imag(c) ≤ tol)
        c = round(real(c), digits = 3)
    end
    c
end
