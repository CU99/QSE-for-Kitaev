#apply creation operator on GS to create first krylov basis state
function creation_op(n,sign,i,j,μ)
    global α = i
    global β = j
    global γ = μ
    D=Dict(0=>I2,1=>X,2=>Y,3=>Z)
    ci=put(n, α=> D[γ]) 
    cj=put(n, β=> D[γ])
    if sign == 1
        cdag = cj
    elseif sign == 2
        cdag = ci
    elseif sign == 3
        cdag = ci+cj
    else
        cdag = ci+im*cj
    end
    return cdag
end

#construct Krylov basis using two-level multigrid time evolution (n_l and n_k may be different to that for the GS)
function create_krylov_basis(Lx, Lz, BC, J, hz,n_l,n_k,n_l2, n_k2, sign,i,j,μ,type)
    n,Jxyz,hmagn,h,ham,ham_norm,kappa=output_ham_properties(Lx, Lz, BC, J, hz,type)
    err,GSE,GS,phi_coeffs,bvs=find_GS_QFD(Lx, Lz, BC, J, hz, n_l, n_k,type)
    println("Check GSE: $(expect(ham,GS))")
    cdag=creation_op(n,sign,i,j,μ)
    psicur= GS|>cdag
    nm=norm(psicur)
    psicur|>normalize!
    println("n_k2,n_l2 =($n_k2, $n_l2)")
    ls = collect(-n_l2:n_l2)
    sizemat = 2*(n_l2+1)*(n_k2+1)-1
    basis_vecs=Vector{ArrayReg}(undef, sizemat)
    kr= collect(-n_k2:n_k2)
    a=1
    for j in 1:length(ls)
        kr= collect(-n_k2:n_k2)
        if ls[j] < 0
            kr= collect(-n_k2:0)
        elseif ls[j] > 0
            kr= collect(0:n_k2)
        end
        for j2 in 1:length(kr)
            tright2 =  2 * pi * kr[j2] / kappa
            tright =  2 * pi * ls[j] * (n_k2+1) / kappa
            psinew = copy(psicur)
            psinew |> time_evolve(ham, tright+tright2)
            basis_vecs[a] = psinew
            a+=1
        end
    end
    return bvs,basis_vecs,phi_coeffs,nm
end

#construct Krylov basis using two-level multigrid time evolution (n_l and n_k may be different to that for the GS), take in GS
function create_krylov_basis(Lx, Lz, BC, J, hz,n_l,n_k,n_l2, n_k2, sign,i,j,μ,GS,phi_coeffs,bvs,type)
    n,Jxyz,hmagn,h,ham,ham_norm,kappa=output_ham_properties(Lx, Lz, BC, J, hz,type)
    cdag=creation_op(n,sign,i,j,μ)
    println("Check GSE: $(expect(ham,GS))")
    psicur= GS|>cdag
    nm=norm(psicur)
    psicur|>normalize!
    println("n_k2,n_l2 =($n_k2, $n_l2)")
    ls = collect(-n_l2:n_l2)
    sizemat = 2*(n_l2+1)*(n_k2+1)-1
    basis_vecs=Vector{ArrayReg}(undef, sizemat)
    kr= collect(-n_k2:n_k2)
    a=1
    for j in 1:length(ls)
        kr= collect(-n_k2:n_k2)
        if ls[j] < 0
            kr= collect(-n_k2:0)
        elseif ls[j] > 0
            kr= collect(0:n_k2)
        end
        for j2 in 1:length(kr)
            tright2 =  2 * pi * kr[j2] / kappa
            tright =  2 * pi * ls[j] * (n_k2+1) / kappa
            psinew = copy(psicur)
            psinew |> time_evolve(ham, tright+tright2)
            basis_vecs[a] = psinew
            a+=1
        end
    end
    return bvs,basis_vecs,phi_coeffs,nm
end

#construct matrices from overlaps of GS and GF basis states
function create_matrices(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2, sign,i,j,μ,type)
    n,Jxyz,hmagn,h,ham,ham_norm,kappa=output_ham_properties(Lx, Lz, BC, J, hz,type)
    phiv,psiv,phi_coeffs,nm = create_krylov_basis(Lx, Lz, BC, J, hz, n_l, n_k,n_l2,n_k2,sign,i,j,μ,type)
    cdag=creation_op(n,sign,i,j,μ)
    sizemat = 2*(n_l2+1)*(n_k2+1)-1
    sizemat2 = 2*(n_l+1)*(n_k+1)-1
    Hmat = Matrix{ComplexF64}(undef, (sizemat, sizemat))
    Smat1 = Matrix{ComplexF64}(undef, (sizemat, sizemat))
    Smat2 = Matrix{ComplexF64}(undef, (sizemat, sizemat2))
    for i=1:sizemat
        psiv1=copy(psiv[i])
        for j=1:sizemat
            psiv2 = copy(psiv[j])
            Smat1[i,j]=statevec(psiv1)'*statevec(psiv2)
            psiv2 |> h
            Hmat[i,j]=statevec(psiv1)'*statevec(psiv2)
        end
    end
    for i=1:sizemat
        psiv1=copy(psiv[i])
        for j=1:sizemat2
            phiv2 = copy(phiv[j])
            phiv2 |> cdag
            Smat2[i,j] = statevec(psiv1)'*statevec(phiv2)
        end
    end
    return Hmat, Smat1, Smat2, phi_coeffs, nm
end



#construct matrices from overlaps of GS and GF basis states (take in GS)
function create_matrices(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2, sign,i,j,μ,GS,phi_coeffs,bvs,type)
    n,Jxyz,hmagn,h,ham,ham_norm,kappa=output_ham_properties(Lx, Lz, BC, J, hz,type)
    phiv,psiv,phi_coeffs,nm = create_krylov_basis(Lx, Lz, BC, J, hz, n_l, n_k,n_l2,n_k2,sign,i,j,μ,GS,phi_coeffs,bvs,type)
    cdag=creation_op(n,sign,i,j,μ)
    sizemat = 2*(n_l2+1)*(n_k2+1)-1
    sizemat2 = 2*(n_l+1)*(n_k+1)-1
    Hmat = Matrix{ComplexF64}(undef, (sizemat, sizemat))
    Smat1 = Matrix{ComplexF64}(undef, (sizemat, sizemat))
    Smat2 = Matrix{ComplexF64}(undef, (sizemat, sizemat2))
    for i=1:sizemat
        psiv1=copy(psiv[i])
        for j=1:sizemat
            psiv2 = copy(psiv[j])
            Smat1[i,j]=statevec(psiv1)'*statevec(psiv2)
            psiv2 |> h
            Hmat[i,j]=statevec(psiv1)'*statevec(psiv2)
        end
    end
    for i=1:sizemat
        psiv1=copy(psiv[i])
        for j=1:sizemat2
            phiv2 = copy(phiv[j])
            phiv2 |> cdag
            Smat2[i,j] = statevec(psiv1)'*statevec(phiv2)
        end
    end
    return Hmat, Smat1, Smat2, phi_coeffs, nm
end


#construct coefficients to build GF
function GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,sign,i,j,μ,type)
    hmat,smat1, smat2, phi_coeffs, nm = create_matrices(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2, sign,i,j,μ,type)
    smat1_inv = pinv(smat1)
    a=[]
    b=[]
    append!(b,0.0)
    psi=Vector{Vector{ComplexF64}}(undef, nkryl+1)
    psi_0 = smat1_inv*smat2*phi_coeffs/nm
    psi[1]=psi_0
    a0 = psi_0'*hmat*psi_0
    append!(a,a0)
    for i=1:nkryl
        println("nkryl=$i")
        b_n2 = abs(psi[i]'*hmat*smat1_inv*hmat*psi[i]-a[i]^2-b[i])
        psi_n = smat1_inv*hmat*psi[i]-a[i]*psi[i]
        if i > 1
            psi_n-=(sqrt(b[i])*psi[i-1])
        end
        psi_n = psi_n/sqrt(b_n2)
        a_n = psi_n'*hmat*psi_n
        append!(a,a_n)
        append!(b,b_n2)
        psi[i+1] = psi_n
    end
    return a,b,psi,nm
end

#construct coefficients to build GF (take in GS)
function GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,sign,i,j,μ,GS,phi_coeffs,bvs,type)
    hmat,smat1, smat2, phi_coeffs, nm = create_matrices(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2, sign,i,j,μ,GS,phi_coeffs,bvs,type)
    smat1_inv = pinv(smat1)
    a=[]
    b=[]
    append!(b,0.0)
    psi=Vector{Vector{ComplexF64}}(undef, nkryl+1)
    psi_0 = smat1_inv*smat2*phi_coeffs/nm
    psi[1]=psi_0
    a0 = psi_0'*hmat*psi_0
    append!(a,a0)
    for i=1:nkryl
        println("nkryl=$i")
        b_n2 = abs(psi[i]'*hmat*smat1_inv*hmat*psi[i]-a[i]^2-b[i])
        psi_n = smat1_inv*hmat*psi[i]-a[i]*psi[i]
        if i > 1
            psi_n-=(sqrt(b[i])*psi[i-1])
        end
        psi_n = psi_n/sqrt(b_n2)
        a_n = psi_n'*hmat*psi_n
        append!(a,a_n)
        append!(b,b_n2)
        psi[i+1] = psi_n
    end
    return a,b,psi,nm
end

function coeffs_to_GF(a,b,nm,z)
    G = 0
    for i = length(a):-1:2
        G=(b[i])/(z-a[i]-G)
    end
    G=1/(z-a[1]-G)
    G*=nm^2
    G2 = 0
    for i = length(a):-1:2
        G2=(b[i])/(z+a[i]-G2)
    end
    G2=1/(z+a[1]-G2)
    G2*=nm^2
    return G,G2
end

#construct GF using continued fraction
function Greens_fn(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,z,sign,i,j,μ,type)
    a,b,psi,nm = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,sign,i,j,μ,type)
    G,G2 = coeffs_to_GF(a,b,nm,z)
    return G, G2
end

#calculate h=/0 GS using imaginary time evolution from h=0 GS
function GS_imag_evol(Lx, Lz, BC, J, hz, type)
    n,Jxyz,hmagn,h,ham,ham_norm,kappa=output_ham_properties(Lx, Lz, BC, J, hz,type)
    w, plaq_seq, vtot = getstabilizers(Lx, Lz, BC, n)
    LXverts, LZverts = getlogicstrings(plaq_seq,Lx,Lz)
    print_initial = false
    #get true ground state energy
    energies, vectors = get_spectrum(h, false)
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
    file = readdlm("C:/Users/cu234/OneDrive - University of Exeter/Projects/Kitaev VQE/Kitaev_VQE_statevec/Kitaev_VQE_statevec/Kitaev code/h=0_GS/N=$(n)_h=0.0_J=$(Jxyz[1])_lr=0.08_depth=$(depth)_niter=500.dat", '\t', Float64, '\n')
    #file = readdlm("./h=0_GS/N=$(n)_h=0.0_J=$(Jxyz[1])_lr=0.08_depth=$(depth)_niter=500.dat", '\t', Float64, '\n')
    params = file[end, :]
    circuit = dispatch!(vqe_centralizer_ansatz(n,depth,Lx,Lz,BC), params)
    ψ = prepare_inital_state(n, num_vortices, plaq_seq, LXverts, LZverts, print_initial, GPU_enabled,Lx,Lz,BC)
    ψ |> circuit
    τ = 1. * n
    #Repeat imaginary time evolution steps
    for i=1:100
        println(i)
        ψ |> time_evolve(h, -1im * τ, tol=1e-12, check_hermicity=false)
        normalize!(ψ)
        EGS = real(expect(h,ψ))
        err = abs(Eground-EGS)
        if err < 10^-12 #if ΔE is at this level, we've already found an accurate GS
            break
        end
    end
    EGS = real(expect(h,ψ))
    err = abs(Eground-EGS)
    println("GS prepared, ΔE=$err")
    return ψ
end

#function to calculate GF using exact diagonalisation (eq 6 in paper)
function check_GF(Lx, Lz, BC, J, hz, z,i,j,μ,type,sign1,sign2)
    n,Jxyz,hmagn,h,ham,ham_norm,kappa=output_ham_properties(Lx, Lz, BC, J, hz,type)
    GS=GS_imag_evol(Lx,Lz,BC,J,hz,type)
    cdag = creation_op(n,sign1,i,j,μ)
    c = creation_op(n,sign2,i,j,μ)
    h2=z*I-Matrix(h)
    h2_inv = inv(h2)
    h3=z*I+Matrix(h)
    h3_inv = inv(h3)
    GF_g = statevec(GS)'*mat(c)*h2_inv*mat(cdag')*statevec(GS)
    GF_l = statevec(GS)'*mat(c')*h3_inv*mat(cdag)*statevec(GS)
    return GF_g, GF_l
end

#Calculates im(Gᵢⱼᵃᵃ) using exact diagonalisation (takes in GS so don't need to calculate each time)
function exact_GF(GS,Lx,Lz,BC,J,hz,z,i,j,μ,type)
    n,Jxyz,hmagn,h,ham,ham_norm,kappa=output_ham_properties(Lx, Lz, BC, J, hz,type)
    cdag = creation_op(n,1,i,j,μ)
    c = creation_op(n,2,i,j,μ)
    h2=z*I-Matrix(h)
    h2_inv = inv(h2)
    h3=z*I+Matrix(h)
    h3_inv = inv(h3)
    GF_g = statevec(GS)'*mat(c)*h2_inv*mat(cdag)*statevec(GS)
    GF_l = statevec(GS)'*mat(c)*h3_inv*mat(cdag)*statevec(GS)
    GF_r = GF_g[1] + GF_l[1]
    return imag(GF_r)
end

#Calculates im(Gᵢⱼᵃᵃ) using QFD (takes in GS)
function QFD_GF(Lx, Lz, BC, J, hz, z, n_l, n_k, n_l2, n_k2,nkryl,i,j,μ,GS,phi_coeffs,bvs,type)
    global α = i
    global β = j
    global γ = μ
    GF = 0
    if i!=j
        GS1 = copy(GS)
        GS2 = copy(GS)
        GS3 = copy(GS)
        ap,bp,psip,nmp = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,3,i,j,μ,GS1,phi_coeffs,bvs,type)
        aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,i,j,μ,GS2,phi_coeffs,bvs,type)
        ai,bi,psii,nmi = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,2,i,j,μ,GS3,phi_coeffs,bvs,type)
        a_cfs = [ap,aj,ai]
        b_cfs = [bp,bj,bi]
        nms = [nmp,nmj,nmi]
        Gp,G2p = coeffs_to_GF(a_cfs[1],b_cfs[1],nms[1],z)
        Gj,G2j = coeffs_to_GF(a_cfs[2],b_cfs[2],nms[2],z)
        Gi,G2i = coeffs_to_GF(a_cfs[3],b_cfs[3],nms[3],z)
        Gc=(Gp-Gj-Gi)/2
        Gc2 = (G2p-G2j-G2i)/2
        GF=Gc+Gc2
    else
        GS1 = copy(GS)
        aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,i,j,μ,GS1,phi_coeffs,bvs,type)
        a_cfs = [aj]
        b_cfs = [bj]
        nms = [nmj]
        Gj,G2j = coeffs_to_GF(a_cfs[1],b_cfs[1],nms[1],z)
        GF=Gj+G2j
    end
    return imag(GF)
end

#Calculates im(Gᵢⱼᵃᵃ) using QFD
function QFD_GF(a_cfs,b_cfs,nms,z,i,j,μ)
    global α = i
    global β = j
    global γ = μ
    GF = 0
    if i!=j
        Gp,G2p = coeffs_to_GF(a_cfs[1],b_cfs[1],nms[1],z)
        Gj,G2j = coeffs_to_GF(a_cfs[2],b_cfs[2],nms[2],z)
        Gi,G2i = coeffs_to_GF(a_cfs[3],b_cfs[3],nms[3],z)
        Gc=(Gp-Gj-Gi)/2
        Gc2 = (G2p-G2j-G2i)/2
        GF=Gc+Gc2
    else
        Gj,G2j = coeffs_to_GF(a_cfs[1],b_cfs[1],nms[1],z)
        GF=Gj+G2j
    end
    return imag(GF)
end



#Calculate DSF using exact diagonalisation
function DSF_exact(Lx,Lz,BC,J,hz,ω,type)
    GS=GS_imag_evol(Lx,Lz,BC,J,hz,type)
    z=ω+0.1im #0.1 or 0.01im might work better
    n, Nx, Nz = gridproperties(Lx, Lz, BC)
    #XXbonds, YYbonds, ZZbonds = edges(Lx, Lz, BC)
    dsf = 0
    for s = 1:n
        for i=1:n
            sx = exact_GF(GS,Lx,Lz,BC,J,hz,z,s,i,1,type) + exact_GF(GS,Lx,Lz,BC,J,hz,z,s,i,2,type) + exact_GF(GS,Lx,Lz,BC,J,hz,z,s,i,3,type)  
            dsf+=sx
        end
    end
    dsf*=1/n
    return dsf
end

#Calculate DSF using exact diagonalisation (nearest-neighbours only)
function DSF_exact_nn(Lx,Lz,BC,J,hz,ω,type)
    GS=GS_imag_evol(Lx,Lz,BC,J,hz,type)
    z=ω+0.1im #0.1 or 0.01im might work better
    n, Nx, Nz = gridproperties(Lx, Lz, BC)
    XXbonds, YYbonds, ZZbonds = edges(Lx, Lz, BC)
    dsf = 0
    for s = 1:n
        for i in XXbonds
            if s in i
                sx = exact_GF(GS,Lx,Lz,BC,J,hz,z,s,i[1],1,type) + exact_GF(GS,Lx,Lz,BC,J,hz,z,s,i[2],1,type)
                dsf+=sx
            end
        end
        for i in YYbonds
            if s in i
                sy = exact_GF(GS,Lx,Lz,BC,J,hz,z,s,i[1],2,type) + exact_GF(GS,Lx,Lz,BC,J,hz,z,s,i[2],2,type)
                dsf+=sy
            end
        end
        for i in ZZbonds
            if s in i
                sz = exact_GF(GS,Lx,Lz,BC,J,hz,z,s,i[1],3,type) + exact_GF(GS,Lx,Lz,BC,J,hz,z,s,i[2],3,type)
                dsf+=sz
            end
        end
    end
    dsf*=1/n
    return dsf
end

#Calculate DSF using exact diagonalisation (nearest-neighbours only) for set of frequencies
function DSF_exact_nn_range_freqs(Lx,Lz,BC,J,hz,ωl,type)
    GS=GS_imag_evol(Lx,Lz,BC,J,hz,type)
    n, Nx, Nz = gridproperties(Lx, Lz, BC)
    XXbonds, YYbonds, ZZbonds = edges(Lx, Lz, BC)
    DSFs=[]
    for ω in ωl
        dsf = 0
        z=ω+0.1im #0.1 or 0.01im might work better
        for s = 1:n
            for i in XXbonds
                if s in i
                    sx = exact_GF(GS,Lx,Lz,BC,J,hz,z,s,i[1],1,type) + exact_GF(GS,Lx,Lz,BC,J,hz,z,s,i[2],1,type)
                    dsf+=sx
                end
            end
            for i in YYbonds
                if s in i
                    sy = exact_GF(GS,Lx,Lz,BC,J,hz,z,s,i[1],2,type) + exact_GF(GS,Lx,Lz,BC,J,hz,z,s,i[2],2,type)
                    dsf+=sy
                end
            end
            for i in ZZbonds
                if s in i
                    sz = exact_GF(GS,Lx,Lz,BC,J,hz,z,s,i[1],3,type) + exact_GF(GS,Lx,Lz,BC,J,hz,z,s,i[2],3,type)
                    dsf+=sz
                end
            end
        end
        dsf*=1/n
        push!(DSFs,dsf)
    end
    return DSFs
end

#Calculate DSF using QFD (exact evolution)
function DSF_QFD(Lx,Lz,BC,J,hz,ω,n_l, n_k, n_l2, n_k2,nkryl,type)
    z=ω+0.1im #0.1 or 0.01im might work better
    n, Nx, Nz = gridproperties(Lx, Lz, BC)
    #XXbonds, YYbonds, ZZbonds = edges(Lx, Lz, BC)
    dsf = 0
    for s = 1:n
        for i=1:n
            if s!=i
                for a=1:3
                    ap,bp,psip,nmp = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,3,s,i,a,type)
                    aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,i,a,type)
                    ai,bi,psii,nmi = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,2,s,i,a,type)
                    acfs = [ap,aj,ai]
                    bcfs = [bp,bj,bi]
                    nms = [nmp,nmj,nmi]
                    sx = QFD_GF(acfs,bcfs,nms,z,s,i,a)
                    dsf+=sx
                end
            else
                for a=1:3
                    aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,i,a,type)
                    acfs = [aj]
                    bcfs = [bj]
                    nms = [nmj]
                    sx = QFD_GF(acfs,bcfs,nms,z,s,i,a)
                    dsf+=sx
                end
            end
        end
    end
    dsf*=1/n
    return dsf
end

#Calculate DSF using QFD (exact evolution), nearest-neighbours only
function DSF_QFD_nn(Lx,Lz,BC,J,hz,ω,n_l, n_k, n_l2, n_k2,nkryl,type)
    z=ω+0.1im #0.1 or 0.01im might work better
    n, Nx, Nz = gridproperties(Lx, Lz, BC)
    XXbonds, YYbonds, ZZbonds = edges(Lx, Lz, BC)
    dsf = 0
    for s = 1:n
        for i in XXbonds
            if s in i
                for j in i
                    if s!=j
                        ap,bp,psip,nmp = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,3,s,j,1,type)
                        aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,j,1,type)
                        ai,bi,psii,nmi = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,2,s,j,1,type)
                        acfs = [ap,aj,ai]
                        bcfs = [bp,bj,bi]
                        nms = [nmp,nmj,nmi]
                        sx = QFD_GF(acfs,bcfs,nms,z,s,j,1)
                        dsf+=sx
                    else
                        aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,j,1,type)
                        acfs = [aj]
                        bcfs = [bj]
                        nms = [nmj]
                        sx = QFD_GF(acfs,bcfs,nms,z,s,j,1)
                        dsf+=sx
                    end
                end
            end
        end
        for i in YYbonds
            if s in i
                for j in i
                    if s!=j
                        ap,bp,psip,nmp = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,3,s,j,2,type)
                        aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,j,2,type)
                        ai,bi,psii,nmi = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,2,s,j,2,type)
                        acfs = [ap,aj,ai]
                        bcfs = [bp,bj,bi]
                        nms = [nmp,nmj,nmi]
                        sx = QFD_GF(acfs,bcfs,nms,z,s,j,2)
                        dsf+=sx
                    else
                        aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,j,2,type)
                        acfs = [aj]
                        bcfs = [bj]
                        nms = [nmj]
                        sx = QFD_GF(acfs,bcfs,nms,z,s,j,2)
                        dsf+=sx
                    end
                end
            end
        end
        for i in ZZbonds
            if s in i
                for j in i
                    if s!=j
                        ap,bp,psip,nmp = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,3,s,j,3,type)
                        aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,j,3,type)
                        ai,bi,psii,nmi = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,2,s,j,3,type)
                        acfs = [ap,aj,ai]
                        bcfs = [bp,bj,bi]
                        nms = [nmp,nmj,nmi]
                        sx = QFD_GF(acfs,bcfs,nms,z,s,j,3)
                        dsf+=sx
                    else
                        aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,j,3,type)
                        acfs = [aj]
                        bcfs = [bj]
                        nms = [nmj]
                        sx = QFD_GF(acfs,bcfs,nms,z,s,j,3)
                        dsf+=sx
                    end
                end
            end
        end
    end
    dsf*=1/n
    return dsf
end

#Calculate DSF using QFD (exact evolution), nearest neighbours for range of frequencies
function DSF_QFD_nn_range_freqs(Lx,Lz,BC,J,hz,ωl,n_l, n_k, n_l2, n_k2,nkryl,type)
    n, Nx, Nz = gridproperties(Lx, Lz, BC)
    XXbonds, YYbonds, ZZbonds = edges(Lx, Lz, BC)
    DSFs=[]
    for ω in ωl
        dsf=0
        z=ω+0.1im #0.1 or 0.01im might work better
        for s = 1:n
            for i in XXbonds
                if s in i
                    for j in i
                        if s!=j
                            ap,bp,psip,nmp = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,3,s,j,1,type)
                            aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,j,1,type)
                            ai,bi,psii,nmi = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,2,s,j,1,type)
                            acfs = [ap,aj,ai]
                            bcfs = [bp,bj,bi]
                            nms = [nmp,nmj,nmi]
                            sx = QFD_GF(acfs,bcfs,nms,z,s,j,1)
                            dsf+=sx
                        else
                            aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,j,1,type)
                            acfs = [aj]
                            bcfs = [bj]
                            nms = [nmj]
                            sx = QFD_GF(acfs,bcfs,nms,z,s,j,1)
                            dsf+=sx
                        end
                    end
                end
            end
            for i in YYbonds
                if s in i
                    for j in i
                        if s!=j
                            ap,bp,psip,nmp = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,3,s,j,2,type)
                            aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,j,2,type)
                            ai,bi,psii,nmi = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,2,s,j,2,type)
                            acfs = [ap,aj,ai]
                            bcfs = [bp,bj,bi]
                            nms = [nmp,nmj,nmi]
                            sx = QFD_GF(acfs,bcfs,nms,z,s,j,2)
                            dsf+=sx
                        else
                            aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,j,2,type)
                            acfs = [aj]
                            bcfs = [bj]
                            nms = [nmj]
                            sx = QFD_GF(acfs,bcfs,nms,z,s,j,2)
                            dsf+=sx
                        end
                    end
                end
            end
            for i in ZZbonds
                if s in i
                    for j in i
                        if s!=j
                            ap,bp,psip,nmp = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,3,s,j,3,type)
                            aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,j,3,type)
                            ai,bi,psii,nmi = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,2,s,j,3,type)
                            acfs = [ap,aj,ai]
                            bcfs = [bp,bj,bi]
                            nms = [nmp,nmj,nmi]
                            sx = QFD_GF(acfs,bcfs,nms,z,s,j,3)
                            dsf+=sx
                        else
                            aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,j,3,type)
                            acfs = [aj]
                            bcfs = [bj]
                            nms = [nmj]
                            sx = QFD_GF(acfs,bcfs,nms,z,s,j,3)
                            dsf+=sx
                        end
                    end
                end
            end
        end
        dsf*=1/n
        push!(DSFs,dsf)
    end
    return DSFs
end

#Calculate DSF using QFD (exact evolution), nearest neighbours for range of frequencies
function DSF_QFD_nn_range_freqs_quick(Lx,Lz,BC,J,hz,ωl,n_l, n_k, n_l2, n_k2,nkryl,type)
    n, Nx, Nz = gridproperties(Lx, Lz, BC)
    XXbonds, YYbonds, ZZbonds = edges(Lx, Lz, BC)
    err,GSE,GS,coeffs,basis_vecs = find_GS_QFD(Lx, Lz, BC, J, hz, n_l, n_k,type)
    DSFs=[]
    for ω in ωl
        dsf=0
        z=ω+0.1im #0.1 or 0.01im might work better
        for s = 1:n
            for i in XXbonds
                if s in i
                    for j in i
                        if s!=j
                            GS1 = copy(GS)
                            GS2 = copy(GS)
                            GS3 = copy(GS)
                            ap,bp,psip,nmp = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,3,s,j,1,GS1,coeffs,basis_vecs,type)
                            aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,j,1,GS2,coeffs,basis_vecs,type)
                            ai,bi,psii,nmi = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,2,s,j,1,GS3,coeffs,basis_vecs,type)
                            acfs = [ap,aj,ai]
                            bcfs = [bp,bj,bi]
                            nms = [nmp,nmj,nmi]
                            sx = QFD_GF(acfs,bcfs,nms,z,s,j,1)
                            dsf+=sx
                        else
                            GS1 = copy(GS)
                            aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,j,1,GS1,coeffs,basis_vecs,type)
                            acfs = [aj]
                            bcfs = [bj]
                            nms = [nmj]
                            sx = QFD_GF(acfs,bcfs,nms,z,s,j,1)
                            dsf+=sx
                        end
                    end
                end
            end
            for i in YYbonds
                if s in i
                    for j in i
                        if s!=j
                            GS1 = copy(GS)
                            GS2 = copy(GS)
                            GS3 = copy(GS)
                            ap,bp,psip,nmp = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,3,s,j,2,GS1,coeffs,basis_vecs,type)
                            aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,j,2,GS2,coeffs,basis_vecs,type)
                            ai,bi,psii,nmi = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,2,s,j,2,GS3,coeffs,basis_vecs,type)
                            acfs = [ap,aj,ai]
                            bcfs = [bp,bj,bi]
                            nms = [nmp,nmj,nmi]
                            sx = QFD_GF(acfs,bcfs,nms,z,s,j,2)
                            dsf+=sx
                        else
                            GS1 = copy(GS)
                            aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,j,2,GS1,coeffs,basis_vecs,type)
                            acfs = [aj]
                            bcfs = [bj]
                            nms = [nmj]
                            sx = QFD_GF(acfs,bcfs,nms,z,s,j,2)
                            dsf+=sx
                        end
                    end
                end
            end
            for i in ZZbonds
                if s in i
                    for j in i
                        if s!=j
                            GS1 = copy(GS)
                            GS2 = copy(GS)
                            GS3 = copy(GS)
                            ap,bp,psip,nmp = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,3,s,j,3,GS1,coeffs,basis_vecs,type)
                            aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,j,3,GS2,coeffs,basis_vecs,type)
                            ai,bi,psii,nmi = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,2,s,j,3,GS3,coeffs,basis_vecs,type)
                            acfs = [ap,aj,ai]
                            bcfs = [bp,bj,bi]
                            nms = [nmp,nmj,nmi]
                            sx = QFD_GF(acfs,bcfs,nms,z,s,j,3)
                            dsf+=sx
                        else
                            GS1 = copy(GS)
                            aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,s,j,3,GS1,coeffs,basis_vecs,type)
                            acfs = [aj]
                            bcfs = [bj]
                            nms = [nmj]
                            sx = QFD_GF(acfs,bcfs,nms,z,s,j,3)
                            dsf+=sx
                        end
                    end
                end
            end
        end
        dsf*=1/n
        push!(DSFs,dsf)
    end
    return DSFs
end


######
#SAME FUNCTIONS JUST WITH TROTTERISATION RATHER THAN EXACT EVOLUTION
######


#construct Krylov basis using two-level multigrid time evolution
#n_l and n_k are for GS, n_l2 and n_k2 are for Krylov basis
function create_krylov_basis_trot(Lx, Lz, BC, J, hz,n_l,n_k,r,n_l2,n_k2,r2,sign,i,j,μ,order,type)
    #find GS
    n,Jxyz,hmagn,h,ham,ham_norm,kappa=output_ham_properties(Lx, Lz, BC, J, hz,type)
    XXbonds, YYbonds, ZZbonds = edges(Lx, Lz, BC)
    err,GSE,GS,phi_coeffs,bvs=find_GS_trot(Lx, Lz, BC, J, hz, n_l,n_k,r,order,type)
    cdag=creation_op(n,sign,i,j,μ)
    psicur= GS|>cdag
    nm=norm(psicur)
    psicur|>normalize!
    #create rest of Krylov basis using two-level multigrid QSE
    println("n_k2,n_l2 =($n_k2, $n_l2)")
    ls = collect(-n_l2:n_l2)
    sizemat = 2*(n_l2+1)*(n_k2+1)-1
    basis_vecs=Vector{ArrayReg}(undef, sizemat)
    kr= collect(-n_k2:n_k2)
    a=1
    for j in 1:length(ls)
        kr= collect(-n_k2:n_k2)
        if ls[j] < 0
            kr= collect(-n_k2:0)
        elseif ls[j] > 0
            kr= collect(0:n_k2)
        end
        for j2 in 1:length(kr)
            tright2 =  2 * pi * kr[j2] / kappa
            tright =  2 * pi * ls[j] * (n_k2+1) / kappa
            psinew = copy(psicur)
            for q=1:abs(ls[j])
                for m=1:r2
                    if ls[j] >0
                        psinew |> trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,tright,r2,ls[j],order,type)
                    else
                        psinew |> trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,tright,r2,ls[j],order,type)'
                    end
                end
            end
            for m=1:r2
                psinew |> trot_circ(n,Jxyz,hmagn,XXbonds,YYbonds,ZZbonds,ham,ham_norm,tright2,r2,1,order,type)
            end
            basis_vecs[a] = psinew
            a+=1
        end
    end
    return bvs,basis_vecs,phi_coeffs,nm
end

#construct matrices from overlaps of GS and GF basis states
function create_matrices_trot(Lx, Lz, BC, J, hz, n_l, n_k, r, n_l2, n_k2, r2, sign,i,j,μ,order,type)
    n,Jxyz,hmagn,h,ham,ham_norm,kappa=output_ham_properties(Lx, Lz, BC, J, hz,type)
    #extract GS basis states, GF basis states and phi_coeffs (need these later)
    phiv,psiv,phi_coeffs,nm = create_krylov_basis_trot(Lx, Lz, BC, J, hz,n_l,n_k,r,n_l2,n_k2,r2,sign,i,j,μ,order,type)
    cdag = creation_op(n,sign,i,j,μ)
    #initialise matrices
    sizemat = 2*(n_l2+1)*(n_k2+1)-1
    sizemat2 = 2*(n_l+1)*(n_k+1)-1
    Hmat = Matrix{ComplexF64}(undef, (sizemat, sizemat))
    Smat1 = Matrix{ComplexF64}(undef, (sizemat, sizemat))
    Smat2 = Matrix{ComplexF64}(undef, (sizemat, sizemat2))
    #calculate matrix elements from overlaps of GF/GS basis states
    for i=1:sizemat
        psiv1=copy(psiv[i])
        for j=1:sizemat
            psiv2 = copy(psiv[j])
            Smat1[i,j]=statevec(psiv1)'*statevec(psiv2)
            psiv2 |> h
            Hmat[i,j]=statevec(psiv1)'*statevec(psiv2)
        end
    end
    for i=1:sizemat
        psiv1=copy(psiv[i])
        for j=1:sizemat2
            phiv2 = copy(phiv[j])
            phiv2 |> cdag
            Smat2[i,j] = statevec(psiv1)'*statevec(phiv2)
        end
    end
    return Hmat, Smat1, Smat2, phi_coeffs, nm
end

#construct coefficients to build GF
function GF_coeffs_trot(Lx, Lz, BC, J, hz, n_l, n_k, r, n_l2, n_k2, r2, nkryl, sign,i,j,μ,order,type)
    hmat,smat1, smat2, phi_coeffs, nm =create_matrices_trot(Lx, Lz, BC, J, hz, n_l, n_k, r, n_l2, n_k2, r2, sign,i,j,μ,order,type)
    smat1_inv = pinv(smat1)
    a=[]
    b=[]
    #calculate coefficients using equations in paper
    append!(b,0.0)
    psi=Vector{Vector{ComplexF64}}(undef, nkryl+1)
    psi_0 = smat1_inv*smat2*phi_coeffs/nm
    psi[1]=psi_0
    a0 = psi_0'*hmat*psi_0
    append!(a,a0)
    for i=1:nkryl
        println("nkryl=$i")
        #In the code sent they used absolute value, so I used it here (though not apparent whether this is really needed)
        b_n2 = abs(psi[i]'*hmat*smat1_inv*hmat*psi[i]-a[i]^2-b[i])
        psi_n = smat1_inv*hmat*psi[i]-a[i]*psi[i]
        if i > 1 #only for ψ² and beyond will there be a ψⁿ⁻² to subtract
            psi_n-=(sqrt(b[i])*psi[i-1])
        end
        psi_n = psi_n/sqrt(b_n2)
        a_n = psi_n'*hmat*psi_n
        append!(a,a_n)
        append!(b,b_n2)
        psi[i+1] = psi_n
    end
    return a,b,psi,nm
end

#construct GF using continued fraction
function Greens_fn_trot(Lx, Lz, BC, J, hz, n_l, n_k, r, n_l2, n_k2, r2, nkryl,z, sign,i,j,μ,order,type)
    a,b,psi,nm = GF_coeffs_trot(Lx, Lz, BC, J, hz, n_l, n_k, r, n_l2, n_k2, r2, nkryl, sign,i,j,μ,order,type)
    G,G2 = coeffs_to_GF(a,b,nm,z)
    return G, G2
end

#Calculate DSF using QFD (exact evolution)
function DSF_QFD_trot(Lx,Lz,BC,J,hz,ω,n_l, n_k, r, n_l2, n_k2, r2, nkryl,order,type)
    GS=GS_imag_evol(Lx,Lz,BC,J,hz,type)
    z=ω+0.1im #0.1 or 0.01im might work better
    n, Nx, Nz = gridproperties(Lx, Lz, BC)
    #XXbonds, YYbonds, ZZbonds = edges(Lx, Lz, BC)
    dsf = 0
    for s = 1:n
        for i=1:n
            if s!=i
                for a=1:3
                    a2p,b2p,psi2p,nm2p = GF_coeffs_trot(Lx, Lz, BC, J, hz, n_l, n_k, r, n_l2, n_k2, r2, nkryl,3,s,i,a,order,type)
                    a2j,b2j,psi2j,nm2j = GF_coeffs_trot(Lx, Lz, BC, J, hz, n_l, n_k, r, n_l2, n_k2, r2, nkryl,1,s,i,a,order,type)
                    a2i,b2i,psi2i,nm2i = GF_coeffs_trot(Lx, Lz, BC, J, hz, n_l, n_k, r, n_l2, n_k2, r2, nkryl,2,s,i,a,order,type)
                    acfs = [ap,aj,ai]
                    bcfs = [bp,bj,bi]
                    nms = [nmp,nmj,nmi]
                    sx = QFD_GF(acfs,bcfs,nms,z,s,i,a)
                    dsf+=sx
                end
            else
                for a=1:3
                    aj,bj,psij,nmj = GF_coeffs_trot(Lx, Lz, BC, J, hz, n_l, n_k,r,n_l2,n_k2,r2,nkryl,1,s,i,a,order,type)
                    acfs = [aj]
                    bcfs = [bj]
                    nms = [nmj]
                    sx = QFD_GF(acfs,bcfs,nms,z,s,i,a)
                    dsf+=sx
                end
            end
        end
    end
    dsf*=1/n
    return dsf
end

#compare exact retarded GF to QSE retarded GF (Gᵢⱼᵃᵃ) for range of ωs
function compare_GFS(Lx, Lz, BC, J, hz, n_l, n_k, r, n_l2, n_k2,r2,nkryl,i,j,μ,ω,order,type)
    println("n_l=$n_l, n_k=$n_k, r=$r, n_l2=$n_l2, n_k2=$n_k2, r2=$r2")
    z = collect(ω).+0.1im #0.1 or 0.01im might work better
    n,Jxyz,hmagn,h,ham,ham_norm,kappa=output_ham_properties(Lx, Lz, BC, J, hz,type)
    GS=GS_imag_evol(Lx,Lz,BC,J,hz,type)
    cdag = creation_op(n,1,i,j,μ)
    c = creation_op(n,2,i,j,μ)
    gfc_r = zeros(0)
    gfc_i = zeros(0)
    gft_r=zeros(0)
    gft_i=zeros(0)
    gfe_r = zeros(0)
    gfe_i = zeros(0)
    if i!=j
        ap,bp,psip,nmp = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,3,i,j,μ,type)
        aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,i,j,μ,type)
        ai,bi,psii,nmi = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,2,i,j,μ,type)
        a2p,b2p,psi2p,nm2p = GF_coeffs_trot(Lx, Lz, BC, J, hz, n_l, n_k, r, n_l2, n_k2, r2, nkryl,3,i,j,μ,order,type)
        a2j,b2j,psi2j,nm2j = GF_coeffs_trot(Lx, Lz, BC, J, hz, n_l, n_k, r, n_l2, n_k2, r2, nkryl,1,i,j,μ,order,type)
        a2i,b2i,psi2i,nm2i = GF_coeffs_trot(Lx, Lz, BC, J, hz, n_l, n_k, r, n_l2, n_k2, r2, nkryl,2,i,j,μ,order,type)
        for k in z
            Gp,G2p = coeffs_to_GF(ap,bp,nmp,k)
            Gj,G2j = coeffs_to_GF(aj,bj,nmj,k)
            Gi,G2i = coeffs_to_GF(ai,bi,nmi,k)
            Gc=(Gp-Gj-Gi)/2
            Gc2 = (G2p-G2j-G2i)/2
            GFc=Gc+Gc2
            append!(gfc_r,real(GFc))
            append!(gfc_i,imag(GFc))
            Gpt,G2pt = coeffs_to_GF(a2p,b2p,nm2p,k)
            Gjt,G2jt = coeffs_to_GF(a2j,b2j,nm2j,k)
            Git,G2it = coeffs_to_GF(a2i,b2i,nm2i,k)
            Gt=(Gpt-Gjt-Git)/2
            Gt2 = (G2pt-G2jt-G2it)/2
            GFt=Gt+Gt2
            append!(gft_r,real(GFt))
            append!(gft_i,imag(GFt))
            h2=k*I-Matrix(h)
            h2_inv = pinv(h2)
            h3=k*I+Matrix(h)
            h3_inv = pinv(h3)
            Ge = state(GS)'*Matrix(c)*h2_inv*Matrix(cdag)*state(GS)
            Ge2 = state(GS)'*Matrix(c)*h3_inv*Matrix(cdag)*state(GS)
            GFe = Ge[1]+Ge2[1]
            append!(gfe_r,real(GFe))
            append!(gfe_i, imag(GFe))
        end
    else
        aj,bj,psij,nmj = GF_coeffs(Lx, Lz, BC, J, hz, n_l, n_k, n_l2, n_k2,nkryl,1,i,j,μ,type)
        a2j,b2j,psi2j,nm2j = GF_coeffs_trot(Lx, Lz, BC, J, hz, n_l, n_k, r, n_l2, n_k2, r2, nkryl,1,i,j,μ,order,type)
        for k in z
            Gj,G2j = coeffs_to_GF(aj,bj,nmj,k)
            GFc=Gj+G2j
            append!(gfc_r,real(GFc))
            append!(gfc_i,imag(GFc))
            Gjt,G2jt = coeffs_to_GF(a2j,b2j,nm2j,k)
            GFt=Gjt+G2jt
            append!(gft_r,real(GFt))
            append!(gft_i,imag(GFt))
            h2=k*I-Matrix(h)
            h2_inv = pinv(h2)
            h3=k*I+Matrix(h)
            h3_inv = pinv(h3)
            Ge = state(GS)'*Matrix(c)*h2_inv*Matrix(cdag)*state(GS)
            Ge2 = state(GS)'*Matrix(c)*h3_inv*Matrix(cdag)*state(GS)
            GFe = Ge[1]+Ge2[1]
            append!(gfe_r,real(GFe))
            append!(gfe_i, imag(GFe))
        end
    end
    return z,gfe_r,gfe_i,gfc_r,gfc_i,gft_r,gft_i
end
