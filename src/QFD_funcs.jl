global α = 1
global β = 2
global γ = 3

#function to estimate kappa using gergorshin circle theorem
function get_kappa(h)
    # m = mat(h)
    m=h
    # println("test")
    max = sum(abs.(m[1,2:end][1]))+real(m[1, 1])
    min = -sum(abs.(m[1,2:end][1]))+real(m[1, 1])
    # println("min = $min, max = $max")
    for i in 2:length(m[1,:])
        R = sum(abs.(m[i,1:end])) - abs(m[i,i])
        # println("step $i")
        # println("R = $R")
        cur_max = R + real(m[i, i])
        if cur_max > max
            max = cur_max
        end
        cur_min = -R + real(m[i, i])
        if cur_min < min
            min = cur_min
        end
        # println("cur_min = $min, cur_max = $max")
    end
    return max - min #Do we need to add Delta? https://en.wikipedia.org/wiki/Gershgorin_circle_theorem
end

#calculate GS basis vectors and coefficients using QFD
function calc_coeffs_and_vecs2(ham, psivec, kappa, n_l, n_k)
    err = []
    imag_rate = []
    println("n_k,n_l =($n_k, $n_l)")
    ls = collect(-n_l:n_l)
    #Initialise empty matrices
    sizemat = 2*(n_l+1)*(n_k+1)-1
    Hmat = Matrix{ComplexF64}(undef, (sizemat, sizemat))
    Smat = Matrix{ComplexF64}(undef, (sizemat, sizemat))
    psicur = psivec[1]
    psicur2 = psivec[1]
    kl= collect(-n_k:n_k)
    kr= collect(-n_k:n_k)
    #Values of lmin and lmax depend on value of k
    for i in 1:length(ls)
        kl= collect(-n_k:n_k)
        if ls[i] < 0
            kl= collect(-n_k:0)
        elseif ls[i] > 0
            kl= collect(0:n_k)
        end
        for i2 in 1:length(kl)
            for j in 1:length(ls)
                kr= collect(-n_k:n_k)
                if ls[j] < 0
                    kr= collect(-n_k:0)
                elseif ls[j] > 0
                    kr= collect(0:n_k)
                end
                for j2 in 1:length(kr)
                    #set values of time for hamiltonian evolution
                    tleft = - 2 * pi * ls[i] * (n_k+1) / kappa
                    tright =  2 * pi * ls[j] * (n_k+1) / kappa
                    tleft2 = - 2 * pi * kl[i2] / kappa
                    tright2 =  2 * pi * kr[j2] / kappa
                    #based on values of k and l, determine position of matrix element 
                    ishifted = i
                    jshifted = j
                    if ls[i] <=0
                        ishifted = (n_k+1)*i-n_k+(i2-1)
                    else
                        ishifted = (n_k+1)*n_l+2*n_k+1+(n_k+1)*(i-n_l-1)-n_k+(i2-1)
                    end
                    if ls[j] <=0
                        jshifted = (n_k+1)*j-n_k+(j2-1)
                    else
                        jshifted = (n_k+1)*n_l+2*n_k+1+(n_k+1)*(j-n_l-1)-n_k+(j2-1)
                    end
                    #apply time evolution to determine matrix elements
                    psinew = copy(psicur)
                    psinew |>  ham |> time_evolve(ham, tleft+tleft2+tright2+tright)
                    Hmat[ishifted, jshifted] = statevec(psicur2)'*statevec(psinew)
                    psinew = copy(psicur)
                    psinew |> time_evolve(ham, tleft+tleft2+tright2+tright)
                    Smat[ishifted, jshifted] = statevec(psicur2)'*statevec(psinew)
                end
            end
        end
    end
    #println("  correct Smat  ", Smat)
    Smat+=I*10^-12
    H2 = Hermitian(Hmat)
    S2 = Hermitian(Smat)
    println(" Smat Pos-def? ", isposdef(S2))
    println(" Hmat Hermitian? ", ishermitian(H2))
    #solve eigenproblem to determine eigenvalues and eigenvectors, 1st one is lowest
    eig_vals, vecs = lalg.lobpcg(Hmat, rand(sizemat, sizemat),Smat,tol=10^(-16),largest=false)
    Efound_min1=eig_vals[1]
    coeffs = vecs[:,1]
    return Efound_min1,coeffs
end

function output_ham_properties(Lx,Lz,BC,J,hz,type)
    #collect lattice properties
    n, Nx, Nz = gridproperties(Lx, Lz, BC)
    XXbonds, YYbonds, ZZbonds = edges(Lx, Lz, BC)
    println("We consider n = ", n, " qubits")
    flush(Core.stdout)
    #calculate hamiltonian properties
    Jxyz = [J,J,J]
    #default: zero magnetic field. Type 1: magnetic field in z direction. Type 2: uniform magnetic field
    hmagn = [0.0,0.0,0.0]
    if type == 1
        hmagn = [0.0, 0.0, hz]
    elseif type == 2
        hmagn = [hz, hz, hz]
    end
    h = kitaev(n, XXbonds, YYbonds, ZZbonds, Jxyz, hmagn)
    ham_norm = norm(mat(h))
    ham = h/ham_norm
    mham = mat(ham)
    #calculate spectrum normalisation factor
    kappa = get_kappa(mham)
    return n,Jxyz,hmagn,h,ham,ham_norm,kappa
end

#takes in lattice properties, calculates hamiltonian, uses QFD to calculate h.=0 GS and GSE
function find_GS_QFD(Lx, Lz, BC, J, hz, n_l, n_k,type)
    n,Jxyz,hmagn,h,ham,ham_norm,kappa=output_ham_properties(Lx, Lz, BC, J, hz, type)
    w, plaq_seq, vtot = getstabilizers(Lx, Lz, BC, n)
    LXverts, LZverts = getlogicstrings(plaq_seq,Lx,Lz)
    print_initial = false
    #get true ground state energy
    energies, vectors = get_spectrum(ham, false)
    Eground = energies[1]
    println("Eground = $Eground")
    num_vortices = 0
    depth=0
    if n == 8
        depth = 1
        num_vortices = 4
    elseif n == 12
        depth = 2
    end
    #retrieve h=0 GS
    #file = readdlm("./h=0_GS/N=$(n)_h=0.0_J=$(Jxyz[1])_lr=0.08_depth=$(depth)_niter=500.dat", '\t', Float64, '\n')
    file = readdlm("../data/N=$(n)_h=0.0_J=$(Jxyz[1])_lr=0.08_depth=$(depth)_niter=500.dat", '\t', Float64, '\n')
    params = file[end, :]
    circuit = dispatch!(vqe_centralizer_ansatz(n,depth,Lx,Lz,BC), params)
    ψ = prepare_inital_state(n, num_vortices, plaq_seq, LXverts, LZverts, print_initial, GPU_enabled,Lx,Lz,BC)
    psivec = [ψ |> circuit]
    #Use QFD to find GSE and GS vectors
    Efound_min1, coeffs = calc_coeffs_and_vecs2(ham, psivec, kappa, n_l, n_k)
    println(real(Efound_min1))
    GSE=Efound_min1*ham_norm
    #find ΔE
    err = abs(real(Efound_min1-Eground)*ham_norm)
    #construct h/=0 GS by summing over all basis states weighted by coeffs
    GS=zero_state(nqubits(psivec[1]))
    psicur = psivec[1]
    a=1
    sizemat = 2*(n_l+1)*(n_k+1)-1
    basis_vecs=Vector{ArrayReg}(undef, sizemat)
    ls=collect(-n_l:n_l)
    for j in 1:length(ls)
        kr= collect(-n_k:n_k)
        if ls[j] < 0
            kr= collect(-n_k:0)
        elseif ls[j] > 0
            kr= collect(0:n_k)
        end
        for j2 in 1:length(kr)
            tright2 =  2 * pi * kr[j2] / kappa
            tright =  2 * pi * ls[j] * (n_k+1) / kappa
            psinew = copy(psicur)
            psinew |> time_evolve(ham, tright+tright2)
            GS+=coeffs[a]*psinew
            basis_vecs[a] = psinew
            a+=1
        end
    end
    GS-=zero_state(nqubits(psivec[1]))
    return err,GSE,GS,coeffs,basis_vecs
end



######
#SAME FUNCTIONS JUST WITH TROTTERISATION RATHER THAN EXACT EVOLUTION
######

Rxx(n,c,t,th) = chain(n,put(c=>H),put(t=>H),control(c,t=>X),put(t=>Rz(th)),control(c,t=>X),put(c=>H),put(t=>H))
Rzz(n,c,t,th) = chain(n,control(c,t=>X),put(n,t=>Rz(th)),control(c,t=>X))
Ryy(n,c,t,th) = chain(n,put(c=>S'),put(t=>S'), put(c=>H),put(t=>H),control(c,t=>X),put(t=>Rz(th)),control(c,t=>X),put(c=>H),put(t=>H),put(c=>S),put(t=>S))


#circuit for 1st or 2nd order trotterisation
function trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,ts,r,l,order,type)
    circ=chain(n)
    if order == 1
        for k = 1:length(XXbonds)
            push!(circ,Rxx(n,XXbonds[k][1],XXbonds[k][2],Jxyz[1]/ham_norm*(ts/l)/r*2))
        end
        if type == 2 #only add if uniform magnetic field
            for k = 1:n
                push!(circ, put(k=>Rx(hmagn[1]/ham_norm*(ts/l)/r*2)))
            end
        end
        for k = 1:length(YYbonds)
            push!(circ,Ryy(n,YYbonds[k][1],YYbonds[k][2],Jxyz[2]/ham_norm*(ts/l)/r*2))
        end
        if type == 2
            for k = 1:n
                push!(circ,put(k=>Ry(hmagn[2]/ham_norm*(ts/l)/r*2)))
            end
        end
        for k = 1:length(ZZbonds)
            push!(circ,Rzz(n,ZZbonds[k][1],ZZbonds[k][2],Jxyz[3]/ham_norm*(ts/l)/r*2))
        end
        for k = 1:n
            push!(circ,put(k=>Rz(hmagn[3]/ham_norm*(ts/l)/r*2)))
        end
    else
        for k = 1:length(XXbonds)
            push!(circ,Rxx(n,XXbonds[k][1],XXbonds[k][2],Jxyz[1]/ham_norm*(ts/l)/r))
        end
        if type == 2 #only add if uniform magnetic field
            for k = 1:n
                push!(circ, put(k=>Rx(hmagn[1]/ham_norm*(ts/l)/r)))
            end
        end
        for k = 1:length(YYbonds)
            push!(circ,Ryy(n,YYbonds[k][1],YYbonds[k][2],Jxyz[2]/ham_norm*(ts/l)/r))
        end
        if type == 2
            for k = 1:n
                push!(circ,put(k=>Ry(hmagn[2]/ham_norm*(ts/l)/r)))
            end
        end
        for k = 1:length(ZZbonds)
            push!(circ,Rzz(n,ZZbonds[k][1],ZZbonds[k][2],Jxyz[3]/ham_norm*(ts/l)/r))
        end
        for k = 1:n
            push!(circ,put(k=>Rz(hmagn[3]/ham_norm*(ts/l)/r*2)))
        end
        for k = length(ZZbonds):-1:1
            push!(circ,Rzz(n,ZZbonds[k][1],ZZbonds[k][2],Jxyz[3]/ham_norm*(ts/l)/r))
        end
        if type == 2
            for k = n:-1:1
                push!(circ,put(k=>Ry(hmagn[2]/ham_norm*(ts/l)/r)))
            end
        end
        for k = length(YYbonds):-1:1
            push!(circ,Ryy(n,YYbonds[k][1],YYbonds[k][2],Jxyz[2]/ham_norm*(ts/l)/r))
        end
        if type == 2
            for k = n:-1:1
                push!(circ,put(k=>Rx(hmagn[1]/ham_norm*(ts/l)/r)))
            end
        end
        for k = length(XXbonds):-1:1
            push!(circ,Rxx(n,XXbonds[k][1],XXbonds[k][2],Jxyz[1]/ham_norm*(ts/l)/r))
        end
    end
    return circ
end

function calc_coeffs_and_vecs_trot(ham,ham_norm,n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,psivec, kappa, n_l, n_k,r,order,type)
    err = []
    imag_rate = []
    println("n_k,n_l =($n_k, $n_l)")
    ls = collect(-n_l:n_l)
    #Initialise empty matrices
    sizemat = 2*(n_l+1)*(n_k+1)-1
    Hmat = Matrix{ComplexF64}(undef, (sizemat, sizemat))
    Smat = Matrix{ComplexF64}(undef, (sizemat, sizemat))
    psicur = psivec[1]
    psicur2 = psivec[1]
    kl= collect(-n_k:n_k)
    kr= collect(-n_k:n_k)
    #Values of lmin and lmax depend on value of k
    for i in 1:length(ls)
        kl= collect(-n_k:n_k)
        if ls[i] < 0
            kl= collect(-n_k:0)
        elseif ls[i] > 0
            kl= collect(0:n_k)
        end
        for i2 in 1:length(kl)
            for j in 1:length(ls)
                kr= collect(-n_k:n_k)
                if ls[j] < 0
                    kr= collect(-n_k:0)
                elseif ls[j] > 0
                    kr= collect(0:n_k)
                end
                for j2 in 1:length(kr)
                    #set values of time for hamiltonian evolution
                    tleft = - 2 * pi * ls[i] * (n_k+1) / kappa
                    tright =  2 * pi * ls[j] * (n_k+1) / kappa
                    tleft2 = - 2 * pi * kl[i2] / kappa
                    tright2 =  2 * pi * kr[j2] / kappa
                    #based on values of k and l, determine position of matrix element 
                    ishifted = i
                    jshifted = j
                    if ls[i] <=0
                        ishifted = (n_k+1)*i-n_k+(i2-1)
                    else
                        ishifted = (n_k+1)*n_l+2*n_k+1+(n_k+1)*(i-n_l-1)-n_k+(i2-1)
                    end
                    if ls[j] <=0
                        jshifted = (n_k+1)*j-n_k+(j2-1)
                    else
                        jshifted = (n_k+1)*n_l+2*n_k+1+(n_k+1)*(j-n_l-1)-n_k+(j2-1)
                    end
                    #apply time evolution to determine matrix elements
                    psinew = copy(psicur)
                    for q=1:abs(ls[j])
                        for m=1:r
                            if ls[j] >0
                                psinew |> trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,tright,r,ls[j],order,type)
                            else
                                psinew |> trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,tright,r,ls[j],order,type)'
                            end
                        end
                    end
                    for m=1:r
                        psinew |> trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,tright2,r,1,order,type)
                    end
                    psinew |> ham
                    for m=1:r
                        psinew |> trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,tleft2,r,1,order,type)
                    end
                    for q=1:abs(ls[i])
                        for m=1:r
                            if ls[i] >0
                                psinew |> trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,tleft,r,ls[i],order,type)
                            else
                                psinew |> trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,tleft,r,ls[i],order,type)'
                            end
                        end
                    end
                    Hmat[ishifted, jshifted] = statevec(psicur2)'*statevec(psinew)
                    psinew = copy(psicur)
                    for q=1:abs(ls[j])
                        for m=1:r
                            if ls[j] >0
                                psinew |> trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,tright,r,ls[j],order,type)
                            else
                                psinew |> trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,tright,r,ls[j],order,type)'
                            end
                        end
                    end
                    for m=1:r
                        psinew |> trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,tright2,r,1,order,type)
                    end
                    for m=1:r
                        psinew |> trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,tleft2,r,1,order,type)
                    end
                    for q=1:abs(ls[i])
                        for m=1:r
                            if ls[i] >0
                                psinew |> trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,tleft,r,ls[i],order,type)
                            else
                                psinew |> trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,tleft,r,ls[i],order,type)'
                            end
                        end
                    end
                    Smat[ishifted, jshifted] = statevec(psicur2)'*statevec(psinew)
                end
            end
        end
    end
    #println("  correct Smat  ", Smat)
    Smat+=I*10^-12
    H2 = Hermitian(Hmat)
    S2 = Hermitian(Smat)
    println(" Smat Pos-def? ", isposdef(S2))
    println(" Hmat Hermitian? ", ishermitian(H2))
    #solve eigenproblem to determine eigenvalues and eigenvectors, 1st one is lowest
    eig_vals, vecs = lalg.lobpcg(Hmat, rand(sizemat, sizemat),Smat,tol=10^(-16),largest=false)
    Efound_min1=eig_vals[1]
    coeffs = vecs[:,1]
    return Efound_min1,coeffs
end

#takes in lattice properties, calculates hamiltonian, uses QFD to calculate h.=0 GS and GSE with trotterisation
function find_GS_trot(Lx, Lz, BC, J, hz, n_l,n_k,r,order,type)
    #extract hamiltonian
    n,Jxyz,hmagn,h,ham,ham_norm,kappa=output_ham_properties(Lx, Lz, BC, J, hz, type)
    XXbonds, YYbonds, ZZbonds = edges(Lx, Lz, BC)
    w, plaq_seq, vtot = getstabilizers(Lx, Lz, BC, n)
    LXverts, LZverts = getlogicstrings(plaq_seq,Lx,Lz)
    print_initial = false
    #get true ground state energy
    energies, vectors = get_spectrum(ham, false)
    Eground = energies[1]
    println("Eground = $Eground")
    num_vortices = 0
    depth=0
    if n == 8
        depth = 1
        num_vortices = 4
    elseif n == 12
        depth = 2
    end
    #retrieve h=0 GS
    #file = readdlm("./h=0_GS/N=$(n)_h=0.0_J=$(Jxyz[1])_lr=0.08_depth=$(depth)_niter=500.dat", '\t', Float64, '\n')
    file = readdlm("../data/N=$(n)_h=0.0_J=$(Jxyz[1])_lr=0.08_depth=$(depth)_niter=500.dat", '\t', Float64, '\n')
    params = file[end, :]
    circuit = dispatch!(vqe_centralizer_ansatz(n,depth,Lx,Lz,BC), params)
    ψ = prepare_inital_state(n, num_vortices, plaq_seq, LXverts, LZverts, print_initial, GPU_enabled,Lx,Lz,BC)
    psivec = [ψ |> circuit]
    #Use QFD to find GSE and GS vectors
    Efound_min1, coeffs = calc_coeffs_and_vecs_trot(ham,ham_norm,n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,psivec, kappa, n_l, n_k,r,order,type)
    println(real(Efound_min1))
    GSE=Efound_min1*ham_norm
    #find ΔE
    err = abs(real(Efound_min1-Eground)*ham_norm)
    #construct h/=0 GS by summing over all basis states weighted by coeffs
    GS=zero_state(nqubits(psivec[1]))
    psicur = psivec[1]
    a=1
    sizemat = 2*(n_l+1)*(n_k+1)-1
    basis_vecs=Vector{ArrayReg}(undef, sizemat)
    ls=collect(-n_l:n_l)
    for j in 1:length(ls)
        kr= collect(-n_k:n_k)
        if ls[j] < 0
            kr= collect(-n_k:0)
        elseif ls[j] > 0
            kr= collect(0:n_k)
        end
        for j2 in 1:length(kr)
            tright2 =  2 * pi * kr[j2] / kappa
            tright =  2 * pi * ls[j] * (n_k+1) / kappa
            psinew = copy(psicur)
            for q=1:abs(ls[j])
                for m=1:r
                    if ls[j] >0
                        psinew |> trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,tright,r,ls[j],order,type)
                    else
                        psinew |> trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,tright,r,ls[j],order,type)'
                    end
                end
            end
            for m=1:r
                psinew |> trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,tright2,r,1,order,type)
            end
            GS+=coeffs[a]*psinew
            basis_vecs[a] = psinew
            a+=1
        end
    end
    GS-=zero_state(nqubits(psivec[1]))
    return err,GSE,GS,coeffs,basis_vecs
end

