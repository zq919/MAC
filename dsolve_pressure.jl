# ------------------------------------------------------------------ 

using SparseArrays  # for SparseMatrixCSC 
using LinearAlgebra 
# using CairoMakie
using Plots  # CairoMakie.plot seems better than Plots.plot 
gr()  # 切换到 GR 后端
# import CairoMakie
using Random  # Random.seed!(123); println(rand(5))
# using ImageView   # need for ImageView.imshow 

# ------------------------ Study Level 1 --------------------------- 
# Study objective: define AVariable and solve Poisson using finite difference

mutable struct AVariable   # Array Variable: a data defined on MeshRct 
    mdata::Array{Float64}  # m=multidim: can be 1D or 2D or 3D array.   
    #
    jac::SparseMatrixCSC{Float64, Int64}
    # jac::Union{SparseMatrixCSC{Float64, Int64},Sparse3D}  
end 

# convert a function to AVariable: 
function AVariable(var::Function, pnts::Tuple)  # pnts::Array{Float64}) pnt类似传入meshgrid网格
    # note that zeros(Int64,0)==[] is true 
    numDims = length(pnts) 
    # @assert numDims==ndims(mvar.mdata)
    @assert numDims==ndims(pnts[1])
    if numDims==1       
        mdat = var.(pnts[1][:]) #.运算表示对每一个节点使用
    elseif numDims==2  
        mdat = var.(pnts[1][:,:], pnts[2][:,:]) 
    elseif numDims==3  
        mdat = var.(pnts[1][:,:,:], pnts[2][:,:,:], pnts[3][:,:,:]) 
    else
        throw(ErrorException("TODO: MVariable constructor for 4D or higher."))
    end  
    @assert length(size(mdat))==numDims 
    # mdisabled::Array{Bool} = []  # mdisabled = fill(false, size(mdat))
    # numElems = prod(size(mdat)); jac = spzeros(nnumElems, numElems) # do not: jac = spzeros(0,0) 
    numElems = prod(size(mdat)); @assert numElems==length(mdat) #确保维数正确, 所有待求节点个数
    jac = spdiagm(ones(numElems)) 
    avar = AVariable(mdat, jac) 
    return avar #.mdat 返回一个 numDims 维度矩阵
end

# # 定义函数和网格
# g(x,y) = x^2 + y^2
# x_grid = range(-1, 1, length=50)
# y_grid = range(-1, 1, length=50)

# # 创建AVariable  
# av = AVariable(g, (x_grid, y_grid))

# # 结果：
# # av.mdata: 50×50矩阵，每个点是x²+y²
# # av.jac: 2500×2500单位矩阵（50×50=2500个网格点）

# convert a constant to AVariable: 
function AVariable(cnst::Float64, pnts::Tuple)  # pnts::Array{Float64}) 
    numDims = length(pnts) 
    @assert numDims==ndims(pnts[1])
    mdat = ones(Float64, size(pnts[1])) * cnst
    @assert length(size(mdat))==numDims 
    # mdisabled::Array{Bool} = []  # mdisabled = fill(false, size(mdat))
    # numElems = prod(size(mdat)); jac = spzeros(nnumElems, numElems) # do not: jac = spzeros(0,0) 
    numElems = prod(size(mdat)); @assert numElems==length(mdat)
    jac = spdiagm(ones(numElems)) 
    avar = AVariable(mdat, jac) 
    return avar 
end

# # 创建在10×10网格上值为0.0的常数场
# x_grid = range(-1, 1, length=10)
# y_grid = range(-1, 1, length=10)
# zero_field = AVariable(0.0, (x_grid, y_grid))

# # 结果：
# # zero_field.mdata: 10×10矩阵，所有元素为0.0
# # zero_field.jac: 100×100单位矩阵

function test_AVariable()
    nx = 10; ny = 10
    #range() 不是一个数组，而是一个 StepRangeLen 或类似的范围对象
    xs = collect(range(0, 10, length=nx+1)); ys = collect(range(0, 10, length=ny+1)) #collect()是 julia 中的一个内置函数，用于返回指定集合或迭代器中所有项目的数组
    # println("range := ", range(0, 10, length=nx+1))
    Xs = [x for x in xs, y in ys];           Ys = [y for x in xs, y in ys] 
    avar = AVariable((x,y) -> x*x+y, (Xs,Ys)) #这是一个匿名函数，计算 $f(x,y) = x^2 + y$, 在每个网格点上计算这个函数值
    return avar
end
#
# aa = test_AVariable() 
# println("aa = ", aa)
function fd_register_unknown!(unknown::AVariable)  # unknown::Union{Vector{AVariable}, NTuple{N, AVariable} where N})
    #注册一个未知量（待求解的变量） ! 表示这个函数会修改输入参数
    # need to write only for registering multiple unknowns (set 2D jac for unknowns): 
    # dof = length(unknown.mdata)
    # unknown.jac = spzeros(dof, dof) # for unknowns: it is spzeros(dofs[iU], sum(dofs))
    # unknown.jac[:,1:end] = spdiagm(ones(dof)) 
    @assert size(unknown.jac,1)==length(unknown.mdata)
    #确保雅可比矩阵的行数等于未知量的数量
    verbose = 1 
    if verbose>=1
        println("To register 1 ", " scalar unknown without any tuned parameters.")
    end    
end
#
function test_fd_register_unknown()
    nx = 10; ny = 10
    xs = collect(range(0, 10, length=nx+1)); ys = collect(range(0, 10, length=ny+1))
    Xs = [x for x in xs, y in ys];           Ys = [y for x in xs, y in ys] 
    avar = AVariable((x,y) -> x*x+y, (Xs,Ys))
    # println("avar = ",avar)
    fd_register_unknown!(avar)
    return (avar = avar)
end
#
# aa = test_fd_register_unknown()

function fd_solve(residual::AVariable)
    matS = residual.jac     # the stiffness matrix A 
    # println("matS = ", matS)
    rhs = -residual.mdata   # the right-hand vector b  
    # println("rhs = ", rhs)
    @assert size(matS,1)==length(rhs) && size(matS,2)==length(rhs) 
    raw_soln = matS \ reshape(rhs, length(rhs))  # rhs[:] will allocate new memory (?) 
    return raw_soln 
end
# TODO: change fd_solve(residual::AVariable) to: 
#       solve_equ(f_avar::Function, avar_init::AVariable; solver_option=(scheme=`direct solver`,) )

function fd_get_solution(avar::AVariable, raw_soln::Vector{Float64}) 
    soln = avar.mdata .+ 0.0
    # println("soln.jac = ",avar.jac)
    soln[:] += avar.jac * raw_soln 
    return soln 
end

#定义运算符
function Base.getindex(avar::AVariable, i::Int64, j::Int64)
    numDims = ndims(avar.mdata) 
    @assert numDims==2 
    (mx,my) = size(avar.mdata)
    i_glb = i + (j-1)*mx #以行为节点顺序   左 右 左 右  一维线性索引
    @assert avar.mdata[i_glb]==avar.mdata[i,j] # note NaN==NaN is false 
    mdat = avar.mdata[i:i,j:j]   # to remain 2D 
    jac = avar.jac[i_glb:i_glb,:]   # to remain 2D sparse matrix 
    return AVariable(mdat, jac) 
end

function Base.setindex!(res::AVariable, avar::AVariable, i::Int64, j::Int64)
    numDims = ndims(res.mdata) 
    @assert numDims==2  # @assert numDims==ndims(avar.mdata) 
    (mx,my) = size(res.mdata)
    i_glb = i + (j-1)*mx 
    @assert res.mdata[i_glb]==res.mdata[i,j] # note NaN==NaN is false 
    res.mdata[i:i,j:j] = avar.mdata[:,:] 
    res.jac[i_glb:i_glb,:] = avar.jac[:,:] # todo: may increase # of non-zero entries, need to clean 0 
    return avar
end

function Base.:+(u::AVariable, v::AVariable) 
    @assert size(u.mdata)==size(v.mdata)
    @assert size(u.jac)==size(v.jac)
    mdisabled::Array{Bool} = []
    return AVariable(u.mdata+v.mdata, u.jac+v.jac) 
end
function Base.:+(u::AVariable, v::Float64) 
    return AVariable(u.mdata.+v, u.jac*1.0) 
end
function Base.:+(v::Float64, u::AVariable) 
    return AVariable(u.mdata.+v, u.jac*1.0) 
end

function Base.:-(u::AVariable, v::AVariable) 
    @assert size(u.mdata)==size(v.mdata)
    @assert size(u.jac)==size(v.jac)
    return AVariable(u.mdata-v.mdata, u.jac-v.jac) 
end
function Base.:-(u::AVariable, v::Float64) 
    return AVariable(u.mdata.-v, u.jac*(+1.0))  
end
function Base.:-(v::Float64, u::AVariable) 
    return AVariable(v.-u.mdata, u.jac*(-1.0))
end

function Base.:*(u::AVariable, v::Float64) 
    return AVariable(u.mdata*v, u.jac*v)  
end
function Base.:*(v::Float64, u::AVariable) 
    return AVariable(u.mdata*v, u.jac*v)  
end

function Base.:/(u::AVariable, v::Float64) 
    return AVariable(u.mdata/v, u.jac/v)  
end

function ex_PoissonEqu_implement1()
    # To solve Poisson's Equation: 
    # -∇⋅(∇p)=f=1 in the domain Ω = [0,1]x[0,1]
    #   B.C.: p = pb = 10 on ∂Ω
    #
    # presBdry = (x,y) -> 10.0;  paraSrc = (x,y) -> 1.0 
    presBdry = (x,y) -> x;  paraSrc = (x,y) -> 0.0 #两个数学函数
    #
    # nx = 2; ny = 2  # the number of cells 
    nx = 1; ny = 1  # the number of cells 
    nx = 10; ny = 10
    xs = collect(range(0, 1, length=nx+1)); ys = collect(range(0, 1, length=ny+1))
    xsC = 0.5 * (xs[1:end-1] + xs[2:end]);  ysC = 0.5 * (ys[1:end-1] + ys[2:end]) # cell-centers
    xsCPad = [xs[1]; xsC; xs[end]];         ysCPad = [ys[1]; ysC; ys[end]];  # padded cellCtrs 左右两侧步长为 h/2
    println("xsCPad := ", xsCPad)
    Xs = [x for x in xsCPad, y in ysCPad];  Ys = [y for x in xsCPad, y in ysCPad] 
    p = AVariable(0.0, (Xs,Ys))  # pressure 
    r = AVariable(0.0, (Xs,Ys))  # residual 
    fd_register_unknown!(p)
    hx = 1.0/nx;  hy = 1.0/ny;  
    for ix = 2:length(xsCPad)-1 
        dx1 = hx; dx2 = hx  
        if ix==2 
            dx1 = hx/2.0 
        elseif ix==(length(xsCPad)-1)
            dx2 = hx/2.0
        end
        for iy = 2:length(ysCPad)-1 
            dy1 = hy; dy2 = hy  
            if iy==2 
                dy1 = hy/2.0 
            elseif iy==(length(ysCPad)-1)
                dy2 = hy/2.0 
            end
            d2pdx2 = ( (p[ix+1,iy] - p[ix,iy])/dx2 - (p[ix,iy] - p[ix-1,iy])/dx1 ) / hx #当dx1 = hx/2 相当于采用边界一阶近似
            d2pdy2 = ( (p[ix,iy+1] - p[ix,iy])/dy2 - (p[ix,iy] - p[ix,iy-1])/dy1 ) / hy
            r[ix,iy] = paraSrc(Xs[ix,iy], Ys[ix,iy]) + d2pdx2 + d2pdy2  
        end        
    end 
    #
    # B.C.: 
    for ix = 1:length(xsCPad)
        for iy = 1:length(ysCPad) 
            if ix==1 || ix==length(xsCPad) || iy==1 || iy==length(ysCPad)
                r[ix,iy] = (p[ix,iy] - presBdry(Xs[ix,iy], Ys[ix,iy]))/hx/hy #没有疑问了 b: 相当于0 - 边界值 A: -1，求出来仍然等于边界值
            end
        end
    end
    #
    raw_soln = fd_solve(r)
    pres_soln = fd_get_solution(p, raw_soln) 
    #
    # println("pres_soln = ", pres_soln)
    return (p, r, pres_soln)

end
#
# (p, r, pres_soln) = ex_PoissonEqu_implement1()

# ------------------------ Study Level 2 --------------------------- 
# Study objective: restrict unknown and residual 

function fd_restrict_unknown!(unknown::AVariable, enabled_in::Union{Array{Bool},BitMatrix}) #enabled_in::Union{Array{Bool},BitMatrix}：布尔数组或位矩阵，标识哪些变量是活跃的
    enabled::Array{Bool} = enabled_in # actually no need it, just be sure 
    # mdisabled: m=multidim: can be 1D or 2D or 3D array. size(mdisabled)=size(mdata) 
    # Assert unknown.jac is an identity matrix: 
    @assert nnz(unknown.jac .!= spdiagm(ones(length(unknown.mdata))))==0  
    @assert size(unknown.jac, 1)==length(unknown.mdata)
    @assert size(unknown.jac, 2)==length(enabled)   
    # enabled[:]：将掩码展开为一维数组

# [:, enabled[:]]：选择所有行，但只保留 enabled 为 true 的列

# 这实际上是从单位矩阵中删除了被约束变量的列
    unknown.jac = unknown.jac[:, enabled[:]]  
    return unknown 
end

function fd_restrict_residual!(residual::AVariable, enabled::Union{Array{Bool},BitMatrix})
    # Note: after residual-restriction, residual.mdata will lose its multi-D info, & reduce to 1D. 
    # mdisabled: m=multidim: can be 1D or 2D or 3D array. size(mdisabled)=size(mdata) 
    # println("In fd_restrict_residual!: size(residual) = ", size(residual), ", size(enabled) = ", size(enabled))
    @assert size(residual.jac, 1)==length(residual.mdata)
    @assert length(enabled)==length(residual.mdata)   
    residual.mdata = residual.mdata[enabled[:]] # enabled[:] 将其展平为 1D：
    residual.jac = residual.jac[enabled[:],:] 
    return residual
end

function ex_PoissonEqu_implement2()
    # To solve Poisson's Equation: 
    # -∇⋅(∇p)=f=1 in the domain Ω = [0,1]x[0,1]
    #   B.C.: p = pb = 10 on ∂Ω
    #
    # presBdry = (x,y) -> 10.0;  paraSrc = (x,y) -> 1.0 
    # presBdry = (x,y) -> 10;  paraSrc = (x,y) -> 0.0 
    # presBdry = (x,y) -> x;  paraSrc = (x,y) -> 0.0 
    presBdry = (x,y) -> x+y;  paraSrc = (x,y) -> 0.0 
    #
    # nx = 2; ny = 2  # the number of cells 
    nx = 1; ny = 1  # the number of cells 
    nx = 5; ny = 5
    # 行数（49）：原始变量的数量

# 列数（25）：活跃（未被约束）变量的数量

# 每列对应一个活跃变量，在该变量位置为1，其他为0 这里被约束的变量指的是边界节点变量
    xs = collect(range(0, 1, length=nx+1)); ys = collect(range(0, 1, length=ny+1))
    xsC = 0.5 * (xs[1:end-1] + xs[2:end]);  ysC = 0.5 * (ys[1:end-1] + ys[2:end]) # cell-centers
    xsCPad = [xs[1]; xsC; xs[end]];         ysCPad = [ys[1]; ysC; ys[end]];  # padded cellCtrs
    Xs = [x for x in xsCPad, y in ysCPad];  Ys = [y for x in xsCPad, y in ysCPad] 
    p = AVariable(0.0, (Xs,Ys))  # pressure 
    println("x1 = ", Xs); println("y1 = ", Ys)
    enabled::Array{Bool} = falses(size(p.mdata)); enabled[2:end-1,2:end-1] .= true 
    fd_restrict_unknown!(p, enabled)
    # println("p2 = ", p) #b: (49,49) A:(49,25)
    #
    fd_register_unknown!(p)
    #
    # B.C.:  
    for ix = 1:length(xsCPad)
        for iy = 1:length(ysCPad) 
            if ix==1 
                p[ix,iy] = 2.0*presBdry(Xs[ix,iy], Ys[ix,iy]) - p[ix+1,iy] 
            elseif ix==length(xsCPad) 
                p[ix,iy] = 2.0*presBdry(Xs[ix,iy], Ys[ix,iy]) - p[ix-1,iy] 
            elseif iy==1 
                p[ix,iy] = 2.0*presBdry(Xs[ix,iy], Ys[ix,iy]) - p[ix,iy+1] 
            elseif iy==length(ysCPad)
                p[ix,iy] = 2.0*presBdry(Xs[ix,iy], Ys[ix,iy]) - p[ix,iy-1] 
            end
            #加不加下面这一行不影响结果, 角点先被约束（从求解变量中移除）   正则化法（添加约束）,强制在一点处等于0
            if (ix==1 || ix==length(xsCPad)) && (iy==1 || iy==length(ysCPad))
                p[ix,iy] = 0.0*p[ix,iy]  #将四个角点的值设为0，这是一种临时的简化处理，通常需要更复杂的角点边界条件。
            end

            # if ix==1 || ix==length(xsCPad) || iy==1 || iy==length(ysCPad) # tmp debug
            #     p[ix,iy] = 1.0*presBdry(Xs[ix,iy], Ys[ix,iy]) + 0.0*p[ix,iy] # tmp debug 
            # end # tmp debug
        end
    end 
    # println("p = ", p)
    #
    r = AVariable(0.0, (Xs,Ys))  # residual 
    #保持残差函数值 r.mdata 不变，只重置导数信息
    r.jac = spzeros(size(r.jac,1), size(p.jac,2))  # avoid creating r: e.g. r = 0.0*p，因为需要分配新内存，进行矩阵乘法 
    hx = 1.0/nx;  hy = 1.0/ny;  
    for ix = 2:length(xsCPad)-1 
        dx1 = hx; dx2 = hx  
        for iy = 2:length(ysCPad)-1 
            dy1 = hy; dy2 = hy  
            d2pdx2 = ( (p[ix+1,iy] - p[ix,iy])/dx2 - (p[ix,iy] - p[ix-1,iy])/dx1 ) / hx
            d2pdy2 = ( (p[ix,iy+1] - p[ix,iy])/dy2 - (p[ix,iy] - p[ix,iy-1])/dy1 ) / hy
            r[ix,iy] = paraSrc(Xs[ix,iy], Ys[ix,iy]) + d2pdx2 + d2pdy2  
        end        
    end 
    #
    fd_restrict_residual!(r, enabled)
    #
    # println("rjac = ", r)
    raw_soln = fd_solve(r)
    pres_soln = fd_get_solution(p, raw_soln) 
    #
    return (p, r, pres_soln)
end
#
# (p, r, pres_soln) = ex_PoissonEqu_implement2()

# ------------------------ Study Level 3 --------------------------- 
# Study objective: vectorization 

function Base.ndims(avar::AVariable) 
    return ndims(avar.mdata)
end
#
function Base.size(avar::AVariable) 
    return size(avar.mdata)
end
#
function Base.size(avar::AVariable, i_dim::Int64) 
    return size(avar.mdata, i_dim)
end

#提取第 i 行，jj 列的数据和雅可比矩阵
function Base.getindex(avar::AVariable, i::Int64, jj::Union{Vector{Int64},UnitRange{Int64}})
    numDims = ndims(avar.mdata) 
    @assert numDims==2 
    (mx,my) = size(avar.mdata)
    ii_glb = collect(i .+ (jj .- 1)*mx)  #类似于多个未知量，但处理形式一样
    @assert avar.mdata[ii_glb]==avar.mdata[i,jj] # note NaN==NaN is false 
    mdat = avar.mdata[i:i,jj]   # to remain 2D 
    jac = avar.jac[ii_glb,:]   # to remain 2D sparse matrix  提取对应的雅可比子矩阵
    return AVariable(mdat, jac) 
end
#提取 ii 行，第 j 列的数据。
function Base.getindex(avar::AVariable, ii::Union{Vector{Int64},UnitRange{Int64}}, j::Int64)
    numDims = ndims(avar.mdata) 
    @assert numDims==2 
    (mx,my) = size(avar.mdata)
    ii_glb = collect(ii .+ (j - 1)*mx)  
    @assert avar.mdata[ii_glb]==avar.mdata[ii,j] # note NaN==NaN is false 
    mdat = avar.mdata[ii,j:j]   # to remain 2D 
    jac = avar.jac[ii_glb,:]   # to remain 2D sparse matrix 
    return AVariable(mdat, jac) 
end
#提取任意矩形区域的数据块
function Base.getindex(avar::AVariable, ii::Union{Vector{Int64},UnitRange{Int64}}, jj::Union{Vector{Int64},UnitRange{Int64}})
    # test: kk = collect(1:4) .+ collect((1:3)*10)'; kk[:] 
    numDims = ndims(avar.mdata) 
    @assert numDims==2 
    (mx,my) = size(avar.mdata)
    ii_glb = collect(ii) .+ ((collect(jj).-1)*mx)'
    ii_glb = ii_glb[:] 
    @assert avar.mdata[ii_glb]==avar.mdata[ii,jj][:] # note NaN==NaN is false 
    mdat = avar.mdata[ii,jj]   # to remain 2D 
    jac = avar.jac[ii_glb,:]   # to remain 2D sparse matrix 
    return AVariable(mdat, jac) 
end

#********************************切片赋值操作****************************
function Base.setindex!(res::AVariable, avar::AVariable, i::Int64, jj::Union{Vector{Int64},UnitRange{Int64}})
    numDims = ndims(res.mdata) 
    @assert numDims==2  # @assert numDims==ndims(avar.mdata) 
    (mx,my) = size(res.mdata)
    ii_glb = collect(i .+ (jj .- 1)*mx)  
    @assert res.mdata[ii_glb]==res.mdata[i,jj] # note NaN==NaN is false 
    res.mdata[i:i,jj] = avar.mdata[:,:] 
    res.jac[ii_glb,:] = avar.jac[:,:] # todo: may increase # of non-zero entries, need to clean 0 
    return avar
end
#
function Base.setindex!(res::AVariable, avar::AVariable, ii::Union{Vector{Int64},UnitRange{Int64}}, j::Int64)
    numDims = ndims(res.mdata) 
    @assert numDims==2  # @assert numDims==ndims(avar.mdata) 
    (mx,my) = size(res.mdata)
    ii_glb = collect(ii .+ (j - 1)*mx)  
    @assert res.mdata[ii_glb]==res.mdata[ii,j] # note NaN==NaN is false 
    res.mdata[ii,j:j] = avar.mdata[:,:] 
    res.jac[ii_glb,:] = avar.jac[:,:] # todo: may increase # of non-zero entries, need to clean 0 
    return avar
end
#
function Base.setindex!(res::AVariable, avar::AVariable, ii::Union{Vector{Int64},UnitRange{Int64}}, jj::Union{Vector{Int64},UnitRange{Int64}})
    numDims = ndims(res.mdata) 
    @assert numDims==2  # @assert numDims==ndims(avar.mdata) 
    (mx,my) = size(res.mdata)
    ii_glb = collect(ii) .+ ((collect(jj).-1)*mx)'
    ii_glb = ii_glb[:] 
    @assert res.mdata[ii_glb]==res.mdata[ii,jj][:] # note NaN==NaN is false 
    res.mdata[ii,jj] = avar.mdata[:,:] #更新数据
    res.jac[ii_glb,:] = avar.jac[:,:] # todo: may increase # of non-zero entries, need to clean 0 
    return avar
end

# Base.axes will be used by Base.getindex(..) and Base.setindex!(..): 
function Base.axes(avar::AVariable, i_dim::Int64) 
    @assert i_dim==1 || i_dim==2 || i_dim==3
    return Base.OneTo(size(avar, i_dim))
end

function Base.:-(u::AVariable, v::Array{Float64})  # diff from Vector{Float64}
    return AVariable(u.mdata - v, u.jac*(+1.0))  
end
function Base.:-(v::Array{Float64}, u::AVariable) 
    return AVariable(v - u.mdata, u.jac*(-1.0))
end


# function kk_test()
#     kk = collect(1:4) .+ collect((1:3)*10)' #在Julai中 “, ”表示换行，空格表示从左往右排序
#     println("k := ",kk)
#     kk_list = copy(kk) #
#     println("klist1 := ",kk_list)
#     return kk_list
# end

# a = kk_test()

#直接赋值, b 和 a 指向同一个内存地址, 修改 b 也会修改 a, 这是引用语义

function ex_PoissonEqu_implement3()
    # To solve Poisson's Equation: 
    # -∇⋅(∇p)=f=1 in the domain Ω = [0,1]x[0,1]
    #   B.C.: p = pb = 10 on ∂Ω
    #
    presBdry = (x,y) -> 10.0;  paraSrc = (x,y) -> 100.0 
    # presBdry = (x,y) -> 10;  paraSrc = (x,y) -> 0.0 
    # presBdry = (x,y) -> x;  paraSrc = (x,y) -> 0.0 
    # presBdry = (x,y) -> x+y;  paraSrc = (x,y) -> 0.0 
    #
    # nx = 2; ny = 2  # the number of cells 
    nx = 1; ny = 1  # the number of cells 
    nx = 5; ny = 5
    xs = collect(range(0, 1, length=nx+1)); ys = collect(range(0, 1, length=ny+1))
    xsC = 0.5 * (xs[1:end-1] + xs[2:end]);  ysC = 0.5 * (ys[1:end-1] + ys[2:end]) # cell-centers
    xsCPad = [xs[1]; xsC; xs[end]];         ysCPad = [ys[1]; ysC; ys[end]];  # padded cellCtrs
    Xs = [x for x in xsCPad, y in ysCPad];  Ys = [y for x in xsCPad, y in ysCPad] 
    p = AVariable(0.0, (Xs,Ys))  # pressure 
    enabled::Array{Bool} = falses(size(p)); enabled[2:end-1,2:end-1] .= true 
    fd_restrict_unknown!(p, enabled)
    #
    fd_register_unknown!(p)
    #
    # B.C.:  
    #p[1,1:end] = 20.0 - p[2,1:end] 相当于鬼点
    p[1,1:end] = 2.0*presBdry.(Xs[1:1,1:end], Ys[1:1,1:end]) - p[2,1:end] 
    p[end,1:end] = 2.0*presBdry.(Xs[end:end,1:end], Ys[end:end,1:end]) - p[end-1,1:end] 
    p[1:end,1] = 2.0*presBdry.(Xs[1:end,1:1], Ys[1:end,1:1]) - p[1:end,2] 
    p[1:end,end] = 2.0*presBdry.(Xs[1:end,end:end], Ys[1:end,end:end]) - p[1:end,end-1] 
    p[[1,end],[1,end]] = 0.0*p[[1,end],[1,end]]
    #
    hx = 1.0/nx;  hy = 1.0/ny;  
    d2pdx2 = ( (p[3:end,2:end-1] - p[2:end-1,2:end-1]) - (p[2:end-1,2:end-1] - p[1:end-2,2:end-1])) / (hx*hx)
    d2pdy2 = ( (p[2:end-1,3:end] - p[2:end-1,2:end-1]) - (p[2:end-1,2:end-1] - p[2:end-1,1:end-2])) / (hy*hy)
    r = paraSrc(Xs[2:end-1,2:end-1], Ys[2:end-1,2:end-1]) + d2pdx2 + d2pdy2  
    #
    # fd_restrict_residual!(r, enabled)  # DO NOT restrict here as residual is already at interior points only. 
    #
    raw_soln = fd_solve(r)
    pres_soln = fd_get_solution(p, raw_soln) 
    #
    return (p, r, pres_soln)
end
#
# (p, r, pres_soln) = ex_PoissonEqu_implement3()


# ------------------------ Study Level 4 --------------------------- 
# Study objective: on an irregular domain and visualize pres and vel 

function save_plot_with_dir(plot_obj, filename; folder="figures", create_dir=true)
    """
    保存图片到指定文件夹，自动创建目录
    """
    # 构建完整路径
#     joinpath("/home/myuser", "example.jl")
# "/home/myuser/example.jl"
    full_path = joinpath(folder, filename)
    
    # 创建目录（如果需要）
    if create_dir
        dir_path = dirname(full_path)
        if !isdir(dir_path)
            mkpath(dir_path)
            println("创建目录: $dir_path")
        end
    end
    
    # 保存图片
    try
        savefig(plot_obj, full_path)
        println("图片已保存: $full_path")
        return true
    catch e
        println("保存失败: $e")
        return false
    end
end


function ex_PoissonEqu_implement4() #此时节点均在计算域内
    # To solve Poisson's Equation: 
    # -∇⋅(∇p)=f in the domain Ω ⊂ Ω0 = [0,1]x[0,1]
    #   B.C.: p = pb on ∂Ω
    #   where Ω = circle((0.5,0.5),r=0.45) \ square(circle((0.5,0.5),side=0.4))
    # # Ω = 圆((0.5,0.5),r=0.45) \ 正方形(中心(0.5,0.5),边长=0.4)
    presBdry = (x,y) -> 10.0;  paraSrc = (x,y) -> 1000.0 
    # presBdry = (x,y) -> 10.0;  paraSrc = (x,y) -> 0.0 
    # presBdry = (x,y) -> x;  paraSrc = (x,y) -> 0.0 
    # presBdry = (x,y) -> x+y;  paraSrc = (x,y) -> 0.0 
    inDomain = (x,y) -> ((x-0.5)^2 + (y-0.5)^2)<0.45^2 && !(abs(x-0.5)<0.2 && abs(y-0.5)<0.2) #判断是否在区域内 要求计算域真包含离散域
    # be sure ∂Ω does NOT overlap with ∂Ω0 
    #
    # nx = 2; ny = 2  # the number of cells 
    nx = 1; ny = 1  # the number of cells 
    nx = 5; ny = 5
    nx = 15; ny = 15
    # nx = 50; ny = 50
    # nx = 50; ny = 60
    # nx = 150; ny = 150
    xs = collect(range(0, 1, length=nx+1)); ys = collect(range(0, 1, length=ny+1)) #0:1/nx+1:1
    xsC = 0.5 * (xs[1:end-1] + xs[2:end]);  ysC = 0.5 * (ys[1:end-1] + ys[2:end]) # cell-centers # no need pad here 
    
    Xs = [x for x in xsC, y in ysC];  Ys = [y for x in xsC, y in ysC] #表示所有中点
    # println("Xs = ",Xs)

    p = AVariable(0.0, (Xs,Ys))  # pressure 
    # res = Vector{AVariable}[]
    # res = [p; p]
    # println("res :=", res)

    # enabled::Array{Bool} = falses(size(p)); enabled[2:end-1,2:end-1] .= true 
    enabled::Array{Bool} = inDomain.(Xs, Ys) # 标记在域内的点

    # p1 = heatmap(enabled, title="org", aspect_ratio=:equal)
    # save_plot_with_dir(p1, "Poinn/org.png", folder="results")

    fd_restrict_unknown!(p, enabled) # 约束域外的点
    #
    fd_register_unknown!(p)
    #
    # step 1: get dpdx and dpdy 
    hx = 1.0/nx;  hy = 1.0/ny;  
    dpdx = (p[2:end,1:end] - p[1:end-1,1:end]) / hx 
    dpdy = (p[1:end,2:end] - p[1:end,1:end-1]) / hy 
    # step 2: correct dpdx on ∂Ω 
    xsEdg = 0.5*(xsC[1:end-1] + xsC[2:end]) #等于内部边 
    # println("xsEdg = ", xsEdg)
    isEdgX0 = enabled[2:end,:] .& (.!enabled[1:end-1,:]) # 当 !enabled[1:end-1,:] 左侧启用 && 右侧禁用 时为 true  左边界 内部下一层，黑边白，白白对应为下边
    isEdgX1 = enabled[1:end-1,:] .& (.!enabled[2:end,:]) #相同块色会变黑，互异块色会变白

    # p2 = heatmap((.!enabled[1:end-1,:]), title="left", aspect_ratio=:equal)
    # save_plot_with_dir(p2, "Poinn/zuoqi.png", folder="results")
    # p3 = heatmap(enabled[2:end,:], title="left", aspect_ratio=:equal)
    # save_plot_with_dir(p3, "Poinn/youqi.png", folder="results")

    # p4 = heatmap(isEdgX0, title="left", aspect_ratio=:equal)
    # save_plot_with_dir(p4, "Poinn/zuobian.png", folder="results")
    # p5 = heatmap(isEdgX1, title="left", aspect_ratio=:equal)
    # save_plot_with_dir(p5, "Poinn/youbian.png", folder="results")

    for i in 1:(length(xsC)-1)
        for j in 1:length(ysC) 
            if isEdgX0[i,j]
                # println("bianjie--caiyong")
                @assert !isEdgX1[i,j] 
                dpdx[i,j] = (p[i+1,j] - presBdry(xsEdg[i],ysC[j])) / (hx/2.0) #N个导数，第一个与最后一个应该是除以h/2, 可以理解为该差分是每个节点的向前差分
            elseif isEdgX1[i,j] 
                # println("bianjie--caiyong2")
                @assert !isEdgX0[i,j]
                dpdx[i,j] = (presBdry(xsEdg[i],ysC[j]) - p[i,j]) / (hx/2.0)
            end 
        end
    end
    # step 3: correct dpdy on ∂Ω 
    ysEdg = 0.5*(ysC[1:end-1] + ysC[2:end])
    isEdgY0 = enabled[:,2:end] .& (.!enabled[:,1:end-1])
    isEdgY1 = enabled[:,1:end-1] .& (.!enabled[:,2:end])
    for i in 1:length(xsC)
        for j in 1:(length(ysC)-1)
            if isEdgY0[i,j]
                @assert !isEdgY1[i,j] 
                dpdy[i,j] = (p[i,j+1] - presBdry(xsC[i],ysEdg[j])) / (hy/2.0)
            elseif isEdgY1[i,j] 
                @assert !isEdgY0[i,j]
                dpdy[i,j] = (presBdry(xsC[i],ysEdg[j]) - p[i,j]) / (hy/2.0)
            end 
        end
    end
    # step 4: get d2pdx2 and d2pdy2    向前差分 - 向后
    d2pdx2 = (dpdx[2:end,2:end-1] - dpdx[1:end-1,2:end-1]) / hx 
    d2pdy2 = (dpdy[2:end-1,2:end] - dpdy[2:end-1,1:end-1]) / hy  
    # step 5: get residual 
    r = paraSrc(Xs[2:end-1,2:end-1], Ys[2:end-1,2:end-1]) + d2pdx2 + d2pdy2  #此时表示所有未求节点，即每个网格的中心 少一个？？？保持怀疑态度
    #
    @assert all(.!enabled[[1,end],:]) && all(.!enabled[:,[1,end]])  # ∂Ω does NOT overlap with ∂Ω0   # 确保物理边界不与计算域边界重叠
    fd_restrict_residual!(r, enabled[2:end-1,2:end-1]) #通过enabled来约束
    #
    raw_soln = fd_solve(r) #牛顿迭代步
    pres_soln = fd_get_solution(p, raw_soln)  #解更新 A(x + p) = b ==> x = A^-1 * (b - Ap)之前求的是这个, 但是完整的x_hat = x + p
    #
    # plot pressure: 
    fig_pres = Plots.contourf(xsC, ysC, pres_soln', color=:viridis, plot_title="pressure solution") 
    # display(fig_pres)
    # savefig(fig_pres, "fig/fig_pres.png")
    # 
    # plot velocity: 
    vx = fd_get_solution(dpdx, raw_soln) * (-1.0);  vx = 0.5*(vx[:,1:end-1] + vx[:,2:end])
    vy = fd_get_solution(dpdy, raw_soln) * (-1.0);  vy = 0.5*(vy[1:end-1,:] + vy[2:end,:]) 
    XsNd = [x for x in xs[2:end-1], y in ys[2:end-1]]; YsNd = [y for x in xs[2:end-1], y in ys[2:end-1]]; 
    @assert size(vx)==size(vy) && size(vx)==size(XsNd) && size(vy)==size(YsNd)     
    magnitudes = sqrt.(vx.^2 + vy.^2) 
    # relativeSizes = magnitudes./maximum(magnitudes) 
    maxArrowLength = 0.2*0.2   # domain [0,1] Umax = 10 then 10 should be ——> 0.2 
    #Only need to define the maxArrowLength to control the entire plot
    scale = maximum(magnitudes)/maxArrowLength # 缩放因子----->箭头
    fig_vel = Plots.quiver(XsNd, YsNd, quiver=(vx/scale, vy/scale), color=:black, plot_title="velocity solution") 
    # display(fig_vel) 
    # 保存图像
    # savefig(fig_vel, "fig/fig_vel1.png")
    #
    return (p, r, pres_soln, fig_pres, fig_vel)
end
#
# (p, r, pres_soln, fig_pres, fig_vel) = ex_PoissonEqu_implement4()  


# ------------------------ Study Level 5 --------------------------- 
# Study objective: multiple unknowns and multiple equations (residuals)
#我们想着需要多个AVariable变量

function aux_flatten(mvars::Union{Vector,Matrix,Tuple,AVariable}) #扁平化处理
    if mvars isa AVariable
        return [mvars] # 单个变量转为单元素数组
    end
    res = Vector{AVariable}[] #T[] 是 Array{T, 1}() 的简写形式，它构造了一个元素类型为 T 的空一维数组
    for mvar in mvars
        if mvar isa AVariable
            res = [res; mvar]  # work for both AVariable and Vector{AVariable} # 添加单个变量
        else
            @assert mvar isa Vector{AVariable} || mvar isa Matrix{AVariable}
            println("work for both AVariable and Vector{AVariable}")
            res = [res; mvar[:]]  # work for both AVariable and Vector{AVariable}
        end
    end
    return res 
end
#恢复结构
function aux_unflatten(flat_mvars::Vector{AVariable}, mvars::Union{Vector,Matrix,Tuple,AVariable})
    if mvars isa AVariable
        @assert length(flat_mvars)==1
        return flat_mvars[1]
    end
    res = Any[] #Any 是所有类型的并集
    k = 1 
    for mvar in mvars
        if mvar isa AVariable
            push!(res, flat_mvars[k]);  k=k+1; #push 在集合中插入一个或多个项目。如果集合是一个有序容器，项目将被插入到最后 (按照给定的顺序)
        else
            @assert mvar isa Vector{AVariable} || mvar isa Matrix{AVariable}  
            item = reshape(flat_mvars[k:k+length(mvar)-1], size(mvar))
            push!(res, item);  k=k+length(mvar) #将扁平化的一维数组恢复为原始的结构
        end
    end
    return res 
end

#注册多变量
function fd_register_unknown!(unknown::Union{Vector,Matrix,Tuple}) #函数名末尾的感叹号 ! 是一个重要的命名约定，它表示这个函数会修改（mutate）其输入参数
    flat_unknowns = aux_flatten(unknown) 
    @assert length(flat_unknowns)>=1 
    #
    # set 2D jac for equ.unknowns: 
    dofs = zeros(Int, length(flat_unknowns))  # 变量的个数
    doqs = zeros(Int, length(flat_unknowns))
    for iU = eachindex(flat_unknowns) #1:length(flat_unknowns)
        # dofs[iU] = length(flat_unknowns[iU].mdata)  # not working if disabling entries 
        doqs[iU] = size(flat_unknowns[iU].jac, 1)  # degrees of quantities (>=dofs) 传入的每个变量自由度
        dofs[iU] = size(flat_unknowns[iU].jac, 2) #列数：自由度
    end
    for iU = eachindex(flat_unknowns)
        @assert flat_unknowns[iU] isa AVariable 
        # @assert is_null(flat_unknowns[iU].jac) || nnz(flat_unknowns[iU].jac)==0
        jac = flat_unknowns[iU].jac 
        flat_unknowns[iU].jac = spzeros(doqs[iU], sum(dofs)) # spzeros(dofs[iU], sum(dofs)) # 创建新的全局雅可比矩阵, 横轴是针对当前变量，纵轴是全局
        i_beg = sum(dofs[1:iU-1])+1;  i_end = sum(dofs[1:iU]) #表示第几个到底几个开始，变量的全局索引
        flat_unknowns[iU].jac[:,i_beg:i_end] = jac  # spdiagm(ones(dofs[iU])), 因为有感叹号存在所以可以修改变量
    end
    #[A 0; 0 B]上述Jac
    #
    verbose = 1 
    if verbose>=1
        if length(flat_unknowns)==1
            println("To register ", length(flat_unknowns), " scalar unknown without any tuned parameters.")
        else
            println("To register ", length(flat_unknowns), " scalar unknowns ", "(", length(unknown), " physical unknowns)", " without any tuned parameters.")
        end
    end    
    # already done above 
end

function fd_solve(residual::Union{Vector,Matrix,Tuple})
    flat_residuals = aux_flatten(residual) 
    for r in flat_residuals
        @assert r isa AVariable 
    end
    matSs = [r.jac for r in flat_residuals]       # the stiffness matrix A  
    matS = sparse_vcat(matSs...)   # 垂直拼接成全局矩阵

    # println("Mat = ",matS)
    rhss = [-r.mdata[:] for r in flat_residuals]  # the right-hand vector b  
    rhs = reduce(vcat, rhss)  # more efficient than vcat(rhss...)  # 垂直拼接
    @assert size(matS,1)==length(rhs) && size(matS,2)==length(rhs) 
    raw_soln = matS \ rhs 
    return raw_soln 
end

# ------------------------ Study Level 6 --------------------------- 
# Study objective: Stokes flow solver 
#递归计算方阵的行列式
function get_determinant(mat::Matrix{Float64})
    @assert size(mat,1)==size(mat,2)
    if size(mat,1)==1
        return mat[1,1]
    elseif size(mat,1)==2
        return mat[1,1]*mat[2,2] - mat[1,2]*mat[2,1]
    elseif size(mat,1)==3
        det1 = get_determinant(mat[2:3,  2:3])
        det2 = get_determinant(mat[[1,3],2:3])
        det3 = get_determinant(mat[1:2,  2:3])
        return mat[1,1]*det1 - mat[2,1]*det2 + mat[3,1]*det3
    else
        println("size(mat) = ", size(mat))
        throw(ErrorException("TODO: get_determinant with size too large"))
    end
end
#计算三角形面积的两倍（有向面积）
function get_detTriangle(pnts::Matrix{Float64}) # twice of its area 
    @assert size(pnts)==(3,2) 
    return get_determinant(hcat([1;1;1], pnts)) 
end

#----------------------------------------------Stokes-------------------------------------------
function solve_StokesFlow(cellType, xsC, ysC, iPresBdry)
    #外围是边界
    #Celltype 包含周围一圈的虚拟中点，如对于ux 1, 2...Nx, Nx+1, 共 Nx+1
    # To solve the Stokes flow: 
    # ∇⋅v = 0 in Ω ⊂ Ω0 = [a0,a1]x[b0,b1]  
    # μ∇⋅∇v - ∇p + f = 0 in Ω ⊂ Ω0  
    #   B.C.: p = pb and u⋅τ=0 on Γₚ ⊂ ∂Ω (the inflow/outflow boundary)
    #         u = 0 (i.e. u⋅n=0 and u⋅τ=0) on Γᵥ ⊂ ∂Ω (the no-flow/no-slip boundary)
    #
    paraMu = 1.0 # viscosity μ 
    paraSrcX = 0.0;  paraSrcY = 0.0  # f = [fx, fy] 
    # paraSrcX = (iC,jC) -> 0.0;  paraSrcY = (iC,jC) -> 0.0  # f = [fx, fy] 
    # be sure ∂Ω does NOT overlap with ∂Ω0 
    #
    # (cellType, xsC, ysC) = ex_get_domain_case2_three_solid_circles()
    # (cellType, xsC, ysC) = ex_get_domain_case1_simple_straignt_channel()
    presBdry = (iC,jC) -> 1.0*(cellType[iC,jC]==iPresBdry) 
    #分配celltype是根据点逆时针连线右侧来判断, 分界线是两点之间的连线

    # presBdry = (iC,jC) -> 100.0*(cellType[iC,jC]==1) 
    # presBdry = (iC,jC) -> 1.0*(cellType[iC,jC]==3) 
    fig_cellType = Plots.contourf(xsC, ysC, cellType', color=:viridis, plot_title="domain cellType"); display(fig_cellType)
    # return fig_cellType 
    hx = xsC[2] - xsC[1];  hy = ysC[2] - ysC[1]  # assuming uniform mesh 
    xsN = vcat(xsC[1]-(hx/2.0), xsC .+ (hx/2.0)) #网格边 所有x剖分边 Nx + 2条，比celltype 中心节点多1
    ysN = vcat(ysC[1]-(hy/2.0), ysC .+ (hy/2.0)) 
    XsC = [x for x in xsC, y in ysC];  YsC = [y for x in xsC, y in ysC]  # for cell center data 
    XsX = [x for x in xsN, y in ysC];  YsX = [y for x in xsN, y in ysC]  # for x-edge center data 
    XsY = [x for x in xsC, y in ysN];  YsY = [y for x in xsC, y in ysN]  # for y-edge center data 
    XsN = [x for x in xsN, y in ysN];  YsN = [y for x in xsN, y in ysN]  # for nodal data 节点数据 共Nx + 2 * Nx + 2
    #
    p = AVariable(0.0, (XsC,YsC))   # pressure 初值以及变量自由度
    enabled::Array{Bool} = (cellType.==0)

    #定义计算变量域内点
    # @assert all(.!enabled[[1,2,end-1,end],:]) && all(.!enabled[:,[1,2,end-1,end]])  # ∂Ω has distance of 2 from ∂Ω0 
    @assert all(.!enabled[[1,end],:]) && all(.!enabled[:,[1,end]])  # ∂Ω has distance of 1 from ∂Ω0 
    fd_restrict_unknown!(p, enabled)
    #
    velx = AVariable(0.0, (XsX,YsX))  # x-component of velocity 
    isEdgX0 = falses(size(XsX)); isEdgX1 = falses(size(XsX)); isEdgX = falses(size(XsX)) #包含虚拟节点边，而之前泊松问题不包含
    isEdgX0[2:end-1,:] = enabled[2:end,:] .& (.!enabled[1:end-1,:])  # boundary x-edges facing west 现在应该属于中间部分，所以横轴要去掉上下进行赋值
    isEdgX1[2:end-1,:] = enabled[1:end-1,:] .& (.!enabled[2:end,:])  # boundary x-edges facing east   
    isEdgX[2:end-1,:]  = enabled[2:end,:] .& enabled[1:end-1,:]      # interior x-edges 
    enabledEgX = isEdgX .| isEdgX0 .| isEdgX1 #此时u_x，u_y虽然包含自己对应的虚拟节点边，但是对应位置的布尔变量是0. 
    fd_restrict_unknown!(velx, enabledEgX) 
    fig_cellType = Plots.heatmap(isEdgX0');
    #             save_plot_with_dir(fig_cellType, "generate_porous_medium/boundary.png", folder="results")
    #
    vely = AVariable(0.0, (XsY,YsY))  # y-component of velocity 
    isEdgY0 = falses(size(XsY)); isEdgY1 = falses(size(XsY)); isEdgY = falses(size(XsY)) #同理包含相应虚拟节点边的索引
    isEdgY0[:,2:end-1] = enabled[:,2:end] .& (.!enabled[:,1:end-1])  # boundary y-edges facing south #所以要
    isEdgY1[:,2:end-1] = enabled[:,1:end-1] .& (.!enabled[:,2:end])  # boundary y-edges facing north 
    isEdgY[:,2:end-1]  = enabled[:,2:end] .& enabled[:,1:end-1]      # interior y-edges 
    enabledEgY = isEdgY .| isEdgY0 .| isEdgY1
    fd_restrict_unknown!(vely, enabledEgY) 
    #
    fd_register_unknown!([p,velx,vely])
    #

    #注意每一项涉及到边界的都需要处理，刚度矩阵单独处理，组装总刚矩阵，届时最终求解时可以删除边界处理

    # Step 1: set the equation for the conservation law of mass (∇⋅v = 0) 
    dvxdx = (velx[2:end,1:end] - velx[1:end-1,1:end]) / hx 
    dvydy = (vely[1:end,2:end] - vely[1:end,1:end-1]) / hy  
    @assert size(dvxdx)==size(p) && size(dvydy)==size(p)  #因为其实对应的是压力的方程
    eqMass = dvxdx + dvydy 
    # Step 2: set the equation for the conservation law of momentum (μ∇⋅∇v - ∇p + f = 0) 
    # 2a) correct to enforce dvxdx=0 and dvydy=0 on inflow/outflow bdry:  
    for i in 1:length(xsC)
        for j in 1:length(ysC) 
            if !enabled[i,j] #bianjie 虚拟中心的导数均为0, u的x方向，v的y方向, 以及颗粒内部
                dvxdx[i,j] = 0.0*velx[i,j]  # <== DD ASSUMPITON HERE for inflow/outflow bdry cond! # 将x方向速度梯度设为0
                dvydy[i,j] = 0.0*vely[i,j]  # <== DD ASSUMPITON HERE for inflow/outflow bdry cond! # 将y方向速度梯度设为0
                # <-- Do not affect for solid-fluid no-slip boundary condition. 
            end
        end
    end
    #
    # 2b) calculate dvxdy and dvydx with bdry treatment:  
    dvxdy =  (velx[1:end,2:end] - velx[1:end,1:end-1]) / hy
    dvydx =  (vely[2:end,1:end] - vely[1:end-1,1:end]) / hx
    for i in 2:length(xsN)-1  # for interior nodes nly  针对待求节点索引 u对y而言，x保持不变,y的中心索引k下面对应的第k条y轴剖分边, 所有索引均从1开始
        for j in 2:length(ysN)-1  
            if cellType[i-1,j-1]>=0 && cellType[i,j-1]>=0 && (cellType[i-1,j]<0 || cellType[i,j]<0) #下边两个单元中心是流体，上边至少有一个是固体
                dvxdy[i,j-1] = (0.0 - velx[i,j-1]) / (hy/2.0)  #velx[i,j-1] 是流体内部距离固体最近点的速度
            elseif cellType[i-1,j]>=0 && cellType[i,j]>=0 && (cellType[i-1,j-1]<0 || cellType[i,j-1]<0)
                dvxdy[i,j-1] = (velx[i,j] - 0.0) / (hy/2.0) 
            elseif cellType[i-1,j]>0 || cellType[i,j]>0 || cellType[i-1,j-1]>0 || cellType[i,j-1]>0 #任何单元是流入流出边界（>0）
                dvxdy[i,j-1] = 0.0*velx[i,j] 
            end
            if cellType[i-1,j-1]>=0 && cellType[i-1,j]>=0 && (cellType[i-1,j]<0 || cellType[i,j]<0)
                dvydx[i-1,j] = (0.0 - vely[i-1,j]) / (hx/2.0) 
            elseif cellType[i,j-1]>=0 && cellType[i,j]>=0 && (cellType[i-1,j-1]<0 || cellType[i-1,j]<0)
                dvydx[i-1,j] = (vely[i,j] - 0.0) / (hx/2.0) 
            elseif cellType[i,j-1]>0 || cellType[i,j]>0 || cellType[i-1,j-1]>0 || cellType[i-1,j]>0
                dvydx[i-1,j] = 0.0*vely[i,j] 
            end
        end
    end
    # 
    # 2c) calculate dpdx and dpdy with bdry treatment (need for inflow/outflow bdry only):  
    
    dpdx = (p[2:end,1:end] - p[1:end-1,1:end]) / hx 
    for i in 2:length(xsN)-1 
        for j in 1:length(ysC) 
            if isEdgX0[i,j]
                @assert !isEdgX1[i,j] 
                # println("pressure = ", presBdry(i-1,j))
                # println("isEdgX0[i,j]: (i,j) = ", (i,j), " cellType[i,j] = ", cellType[i,j], " cellType[i-1,j] = ", cellType[i-1,j])  # dbg
                dpdx[i-1,j] = (p[i,j] - presBdry(i-1,j)) / (hx/2.0) #都是向前, 压力求导在剖分边界处
            elseif isEdgX1[i,j] 
                @assert !isEdgX0[i,j]
                # println("isEdgX1[i,j]: (i,j) = ", (i,j), " cellType[i,j] = ", cellType[i,j], " cellType[i-1,j] = ", cellType[i-1,j])  # dbg
                # println(" presBdry(i,j) = ", presBdry(i,j)) # dbg 
                dpdx[i-1,j] = (presBdry(i,j) - p[i-1,j]) / (hx/2.0)
            end 
        end
    end
    dpdy = (p[1:end,2:end] - p[1:end,1:end-1]) / hy 
    for i in 1:length(xsC) 
        for j in 2:length(ysN)-1 
            if isEdgY0[i,j]
                @assert !isEdgY1[i,j] 
                dpdy[i,j-1] = (p[i,j] - presBdry(i,j-1)) / (hy/2.0)
            elseif isEdgY1[i,j] 
                @assert !isEdgY0[i,j]
                dpdy[i,j-1] = (presBdry(i,j) - p[i,j-1]) / (hy/2.0)
            end 
        end
    end
    #
    # 2d) calculate eqMomtX and eqMomtY with bdry treatment 
    dvxdxx = (dvxdx[2:end,1:end] - dvxdx[1:end-1,1:end]) / hx  # no need for dvxdxy and dvxdyx 
    dvxdyy = (dvxdy[1:end,2:end] - dvxdy[1:end,1:end-1]) / hy 
    dvydxx = (dvydx[2:end,1:end] - dvydx[1:end-1,1:end]) / hx  
    dvydyy = (dvydy[1:end,2:end] - dvydy[1:end,1:end-1]) / hy 
    eqMomtX = paraMu * (dvxdxx[1:end,2:end-1] + dvxdyy[2:end-1,1:end]) - dpdx[1:end,2:end-1] + paraSrcX  
    eqMomtY = paraMu * (dvydxx[1:end,2:end-1] + dvydyy[2:end-1,1:end]) - dpdy[2:end-1,1:end] + paraSrcY  
    #
    # 2e) correct for the no-slip boundaries: 
    @assert size(isEdgX)==(length(xsN), length(ysC))
    for i in 2:length(xsN)-1 
        for j in 2:length(ysC)-1  
            if isEdgX0[i,j] 
                @assert cellType[i,j]==0 && cellType[i-1,j]!=0
                if cellType[i-1,j]<0
                    eqMomtX[i-1,j-1] = velx[i,j] - 0.0  # velx = 0.0 on no-slip edgeX  都减1表示外围边界 通过直接设置动量方程来强制速度为零
                end
            end 
            if isEdgX1[i,j] 
                @assert cellType[i-1,j]==0 && cellType[i,j]!=0
                if cellType[i,j]<0
                    eqMomtX[i-1,j-1] = velx[i,j] - 0.0  # velx = 0.0 on no-slip edgeX  
                end
            end 
        end
    end
    @assert size(isEdgY)==(length(xsC), length(ysN)) #颗粒无滑移边界条件
    for i in 2:length(xsC)-1  
        for j in 2:length(ysN)-1 
            if isEdgY0[i,j] 
                @assert cellType[i,j]==0 && cellType[i,j-1]!=0
                if cellType[i,j-1]<0
                    eqMomtY[i-1,j-1] = vely[i,j] - 0.0  # vely = 0.0 on no-slip edgeY  
                end
            end 
            if isEdgY1[i,j] 
                @assert cellType[i,j-1]==0 && cellType[i,j]!=0
                if cellType[i,j]<0
                    eqMomtY[i-1,j-1] = vely[i,j] - 0.0  # vely = 0.0 on no-slip edgeY  b = 0 - bianjiezhi, A = -1
                end
            end             
        end
    end
    # 
    # println("size(enabled) = ", size(enabled)) # dbg 
    # println("size(enabledEgX) = ", size(enabledEgX)) # dbg 
    # println("size(enabledEgY) = ", size(enabledEgY)) # dbg 
    fd_restrict_residual!(eqMass, enabled) 
    fd_restrict_residual!(eqMomtX, enabledEgX[2:end-1,2:end-1])  #为什么这样，是因为离散区域包含物理计算区域，删除这部分不影响刚度矩阵
    fd_restrict_residual!(eqMomtY, enabledEgY[2:end-1,2:end-1]) 
    #
    raw_soln = fd_solve([eqMass, eqMomtX, eqMomtY])
    p_soln = fd_get_solution(p, raw_soln) 
    vx_soln = fd_get_solution(velx, raw_soln); vy_soln = fd_get_solution(vely, raw_soln);   
    #
    # calculate total flow rate in each of the inlets/outlets, 因为现在是Stokes 方程而不是Darcy, 所以考虑u, v即可
    numInOutLets = maximum(cellType) #返回流入流出max个类型编号，创建max x 1的数组用于储存, 流入流出是根据设计边界来的而不是根据左右猜测
    flowOutRates = zeros(numInOutLets)
    for i in 2:length(xsN)-1 
        for j in 2:length(ysC)-1  
            if isEdgX0[i,j] 
                if cellType[i-1,j]>0
                    flowOutRates[cellType[i-1,j]] += -vx_soln[i,j]*hy #流出流量 = 速度 × 横截面积 西向边界的正向速度是向右，但流出是向左，所以取负
                end #流入应该等于左侧单元的流出，所以往左平移1格
            end 
            if isEdgX1[i,j] 
                if cellType[i,j]>0
                    flowOutRates[cellType[i,j]] += vx_soln[i,j]*hy 
                end
            end 
        end
    end
    for i in 2:length(xsC)-1  
        for j in 2:length(ysN)-1 
            if isEdgY0[i,j] 
                if cellType[i,j-1]>0
                    flowOutRates[cellType[i,j-1]] += -vy_soln[i,j]*hx 
                end
            end 
            if isEdgY1[i,j] 
                if cellType[i,j]>0
                    flowOutRates[cellType[i,j]] += vy_soln[i,j]*hx 
                end
            end             
        end
    end 
    # 
    # plot pressure: 
    fig_pres = Plots.contourf(xsC, ysC, p_soln', color=:viridis, plot_title="pressure solution") 
    display(fig_pres)
    # 
    # plot velocity: 
    vx = 0.5*(vx_soln[2:end-1,1:end-1] + vx_soln[2:end-1,2:end]) #网格节点处
    vy = 0.5*(vy_soln[1:end-1,2:end-1] + vy_soln[2:end,2:end-1]) 
    XsNd = XsN[2:end-1,2:end-1]; YsNd = YsN[2:end-1,2:end-1]
    @assert size(vx)==size(vy) && size(vx)==size(XsNd) && size(vy)==size(YsNd)  
    
   # 调节稀疏性：步长越大越稀疏
    stride = 4  # 尝试 3-8 之间的值
    # 创建降采样索引
    i_idx = 1:stride:size(XsNd, 1)
    j_idx = 1:stride:size(XsNd, 2)
    # 应用降采样
    XsNd_sub = XsNd[i_idx, j_idx]
    YsNd_sub = YsNd[i_idx, j_idx]
    vx_sub = vx[i_idx, j_idx]
    vy_sub = vy[i_idx, j_idx]

    magnitudes = sqrt.(vx_sub.^2 + vy_sub.^2) 
    # relativeSizes = magnitudes./maximum(magnitudes) 
    maxArrowLength = 0.2*0.2   # domain [0,1] Umax = 10 then 10 should be ——> 0.2 
    #Only need to define the maxArrowLength to control the entire plot
    scale = maximum(magnitudes)/maxArrowLength
    fig_vel = Plots.quiver(XsNd_sub, YsNd_sub, quiver=(vx_sub/scale, vy_sub/scale), color=:black, plot_title="velocity solution") 
    # display(fig_vel) 
    # save_plot_with_dir(fig_vel, "generate_porous_medium/fig_vel_$iPresBdry.png", folder="results")
    #
    # dvxdyy=dvxdyy, dvxdy=dvxdy, dpdx=dpdx,  # tmp return 
    return (flowOutRates=flowOutRates, raw_soln=raw_soln, cellType=cellType, enabled=enabled, enabledEgX=enabledEgX, enabledEgY=enabledEgY, 
    p=p, velx=velx, vely=vely, eqMass=eqMass, eqMomtX=eqMomtX, eqMomtY=eqMomtY, p_soln=p_soln, vx_soln=vx_soln, vy_soln=vy_soln, 
    fig_pres=fig_pres, fig_vel=fig_vel, XsN = XsN, YsN = YsN)      
end


function ex_StokesFlow() 
    #此时其实只是计算了其中一块，而我们是多块一起讨论.
    (cellType, xsC, ysC) = ex_get_domain_case2_three_solid_circles()
    # (cellType, xsC, ysC) = ex_get_domain_case1_simple_straignt_channel()
    numInOutLets = maximum(cellType) 
    flows = Vector(undef, numInOutLets) #创建了一个未初始化的向量
    println("flows = ",flows)
    for iPresBdry = 1:numInOutLets 
        println("iPresBdry = ",iPresBdry)
        flows[iPresBdry] = solve_StokesFlow(cellType, xsC, ysC, iPresBdry) 
    end
    matrixFlowOut = hcat([flows[i].flowOutRates for i=1:numInOutLets]...) #hcat 将多个数组水平拼接（按列拼接）
    println("\n\n matrixFlowOut = ", matrixFlowOut, "\n\n")
    return (matrixFlowOut=matrixFlowOut, flows=flows)
end
# res = ex_StokesFlow();

# ------------------------ PNM Application ------------------------- 
# PNM Application 
#cellType: 正值储存Div三角编号，表明所属单元号elem，负值表示颗粒，序号表明第几个颗粒圆, 输出的是严格计算区域，如果想计算，则需要进行扩展
function generate_porous_medium(nx_fin, ny_fin, numCirs, rho)
    #
    # Step 1: to generate the circles that represents porous medium  生成随机圆形颗粒
    # Step Input: domSqCore, cir_radiusRange;  Step Output: circles
    #
    domSqCore = [0.0 0.0; 1.0 1.0]  # domain square defined by [a0 b0; a1 b1] 
    cir_radiusRange = [0.05, 0.12]     # .*5 # 颗粒半径范围
    domSq = zeros(2,2) # padded domSq 扩展计算域（添加边界缓冲）
    domSq[1,:] = domSqCore[1,:] - (domSqCore[2,:] - domSqCore[1,:])*0.2 
    domSq[2,:] = domSqCore[2,:] + (domSqCore[2,:] - domSqCore[1,:])*0.2 
    nx = 100; ny = 100  # only for plotting  
    xs = collect(range(domSq[1,1], domSq[2,1], length=nx+1))
    ys = collect(range(domSq[1,2], domSq[2,2], length=ny+1))
    xsC = 0.5 * (xs[1:end-1] + xs[2:end]);  ysC = 0.5 * (ys[1:end-1] + ys[2:end]) # cell-centers # no need pad here 
    XsC = [x for x in xsC, y in ysC];  YsC = [y for x in xsC, y in ysC] 
    #
    Random.seed!(12345)  # 设置随机种子确保可重现
    circles = zeros(numCirs,3)  # (x,y) and r; later size(circles,1) will decrease 
    cellType = zeros(Int, size(XsC)) 
    kC = 1
    for k = 1:numCirs  # while kC <= numCirs 
        cir_c = domSq[1,:] + (domSq[2,:] - domSq[1,:]) .* rand(2)  # ctr pnt (x,y) 随机圆心
        cir_r = cir_radiusRange[1] + (cir_radiusRange[2] - cir_radiusRange[1]) * rand() # r 半径
        circles[kC,1:2] = cir_c; circles[kC,3] = cir_r #记录数据
        isAway = ((circles[1:kC,1].-cir_c[1]).^2 + (circles[1:kC,2].-cir_c[2]).^2) .> (rho.*(circles[1:kC,3].+cir_r).^2) 
        #1.1 是一个安全系数（safety factor），用于确保生成的圆形颗粒之间保持一定的微小间隙，如果网格太大应该取大一点

        # doesShow = (<10) || (k<100 && k%10==0) || (k<1000 && k%100==0) || (k<10000 && k%1000==0) || k%10000==0 
        doesShow = (kC<10) || (kC<100 && kC%10==0) || (kC<1000 && kC%100==0) || (kC<10000 && kC%1000==0) || kC%10000==0 
        if doesShow 
            println("k = ", k, ", kC = ", kC, "; isAway = ", isAway) 
        end

        if all(isAway[1:end-1]) #如果所有的都成立
            cellType[(XsC.-cir_c[1]).^2 + (YsC.-cir_c[2]).^2 .< cir_r^2] .= -kC # 标记为负值  
            kC += 1 
            if doesShow 
                # println(" --> take circles[kC,:] = ", circles[kC,:])
                fig_cellType = Plots.heatmap(cellType'); display(fig_cellType)
            end
        end 

            if k == numCirs
                fig_cellType = Plots.heatmap(cellType');
                save_plot_with_dir(fig_cellType, "generate_porous_medium/fig_cellType.png", folder="results")
            end
    end 
    circles = circles[1:kC-1,:]
    fig_cellType = Plots.heatmap(cellType'); display(fig_cellType)
    # return (circles=circles, cellType=cellType,)
    #
    # Step 2: to generate the Voronoi diagram defined by the centers of all circles
    # Step Input: domSq(Pad), circles[:,1:2] (but not radius circles[:,3]), rMax, nx, ny   
    # Step Output: cellVoronoi[:,:] (=1,2,... for neigb of ctr 1,2,...); elems[numTriangles/Quads][numPnts] 
    #
    rMax = cir_radiusRange[2]*2.5 # *1.5  # half of estimated max distance btw any two ctr pnts 
    nx = 200; ny = 200   # for resolving and plotting only 
    # domSq = [-0.2 -0.2; 1.2 1.2]
    xs = collect(range(domSq[1,1], domSq[2,1], length=nx+1))
    ys = collect(range(domSq[1,2], domSq[2,2], length=ny+1))
    xsC = 0.5 * (xs[1:end-1] + xs[2:end]);  ysC = 0.5 * (ys[1:end-1] + ys[2:end]) # cell-centers # no need pad here 
    XsC = [x for x in xsC, y in ysC];  YsC = [y for x in xsC, y in ysC] 
    maxDistSq = Inf # ( (xs[end]-xs[1])^2 + (ys[end]-ys[1])^2) * 10000.0  
    cellVoronoi = zeros(Int, size(XsC)) # Voronoi区域标记数组
    distVoronoi = ones(size(XsC)) .* maxDistSq # 最小距离数组
    for kC = 1:size(circles,1)
        pnt = circles[kC,1:2] 
        i0 = max(sum(xsC .< (pnt[1] - rMax)), 1)  #使用 sum(xsC .< value) 来找到对应的网格索引 因为如果成立会为1，所以求和相当于位置索引
        i1 = min(sum(xsC .< (pnt[1] + rMax)), length(xsC))  
        j0 = max(sum(ysC .< (pnt[2] - rMax)), 1) 
        j1 = min(sum(ysC .< (pnt[2] + rMax)), length(ysC))  
        distSq = (XsC[i0:i1,j0:j1] .- pnt[1]).^2 + (YsC[i0:i1,j0:j1] .- pnt[2]).^2 #计算局部窗口内每个网格点到当前颗粒中心的距离平方
        isNeig = (distSq .< distVoronoi[i0:i1,j0:j1])
        cellVoronoi[i0:i1,j0:j1][isNeig] .= kC  # wrong  将更近的点标记为属于当前颗粒的Voronoi区域 .= kC 赋值操作修改的是副本的值
        loc_cellVoronoi = cellVoronoi[i0:i1,j0:j1] # 提取副本
        loc_cellVoronoi[isNeig] .= kC  # 修改副本 类似局部单元编号, 颗粒编号
        cellVoronoi[i0:i1,j0:j1] = loc_cellVoronoi # 写回原数组-编号
        # distVoronoi[i0:i1,j0:j1][isNeig] = distSq[isNeig]  # wrong 
        loc_distVoronoi = distVoronoi[i0:i1,j0:j1]
        loc_distVoronoi[isNeig] = distSq[isNeig] #distSq相当于中间变量 局部单元距离阵
        distVoronoi[i0:i1,j0:j1] = loc_distVoronoi #全局距离矩阵
        doesShow = (kC<10) || (kC<100 && kC%10==0) || (kC<1000 && kC%100==0) || (kC<10000 && kC%1000==0) || kC%10000==0 
        if doesShow 
            println("kC = ", kC, "; maximum(cellVoronoi) = ", maximum(cellVoronoi)) 
            fig_cellVoronoi = Plots.heatmap(cellVoronoi'); display(fig_cellVoronoi)
        end

        if kC == size(circles,1)
            fig_cellVoronoi = Plots.heatmap(cellVoronoi');
            save_plot_with_dir(fig_cellVoronoi, "generate_porous_medium/cellVoronoi.png", folder="results")
        end
    end
    @assert all(cellVoronoi .!= 0)
    ## 比较每个单元(此时节点是中点)左侧的上下两个点 (形成一条垂直边)从原始网格上出发则
    x0 = cellVoronoi[1:end-1,1:end-1] .!= cellVoronoi[2:end,1:end-1] #比较每个网格点与其右侧邻居的Voronoi标记 true: 标记不同 → 存在垂直边界
    x1 = cellVoronoi[1:end-1,2:end]   .!= cellVoronoi[2:end,2:end]
    y0 = cellVoronoi[1:end-1,1:end-1] .!= cellVoronoi[1:end-1,2:end]
    # 比较每个单元下方的左右两个点 (形成另一条水平边)
    y1 = cellVoronoi[2:end,1:end-1]   .!= cellVoronoi[2:end,2:end] 
    numsNeig = x0 + x1 + y0 + y1 #计算每个网格单元边界的交叉数量
    @assert all(numsNeig .!= 1) # can be only 0, 2, 3, or 4 
    # elems = [unique(cellVoronoi[(0:1).+i,(0:1).+j]) for i=1:size(cellVoronoi,1)-1, j=1:size(cellVoronoi,2)-1][numsNeig .>= 3] 
    # counter-clockwise arranged: 
    #提取Voronoi图的顶点信息，这些顶点对应着Delaunay三角剖分中三角形的外接圆圆心点 系统性地获取每个网格单元四个角点的Voronoi标记
    cv1 = cellVoronoi[1:end-1,1:end-1]; cv2 = cellVoronoi[2:end,1:end-1]; 
    cv3 = cellVoronoi[2:end,2:end];     cv4 = cellVoronoi[1:end-1,2:end]; 
    #捕获一个单元内所有涉及的 Voronoi 单元（即 Delaunay 顶点）
    elems = [unique([cv1[i,j],cv2[i,j],cv3[i,j],cv4[i,j]]) for i=1:size(cv1,1), j=1:size(cv1,2)][numsNeig .>= 3] #B = numsNeig .>= 3; C = A(B);
    # return (circles=circles, cellType=cellType, cellVoronoi=cellVoronoi, elems=elems) 
    #
    # Step 3: to generate the Delaunay triangulation diagram  
    # Step Input: domSq(Pad), circles[:,1:2] (but not radius circles[:,3]), elems, nx, ny   
    # Step Output: cellDelaunay[:,:] (=1,2,... for triangle label 1,2,...) 
    #
    
    # nx = 420; ny = 420   # for final resolution  (400 for pad)
    xs = collect(range(domSq[1,1], domSq[2,1], length=nx_fin+1))
    ys = collect(range(domSq[1,2], domSq[2,2], length=ny_fin+1))
    xsC = 0.5 * (xs[1:end-1] + xs[2:end]);  ysC = 0.5 * (ys[1:end-1] + ys[2:end]) # cell-centers # no need pad here 
    XsC = [x for x in xsC, y in ysC];  YsC = [y for x in xsC, y in ysC] 
    cellDelaunay = zeros(Int, size(XsC)) # 输出矩阵，初始为0（未分配）

    Num_Delaunay = size(elems,1)

    for kE = 1:size(elems,1) 
        kPnts = elems[kE] #储存区域单元jiedian编号
        isInRight = (x,y,iP,jP) -> get_detTriangle(vcat(circles[kPnts[[iP,jP]],1:2],[x y])) < 0
        #计算包围盒 并向外扩展2个像素作为安全边际
        i0 = max(sum(xsC .< minimum(circles[kPnts,1]))-2, 1) #circles储存颗粒中心坐标
        i1 = min(sum(xsC .< maximum(circles[kPnts,1]))+2, length(xsC))  
        j0 = max(sum(ysC .< minimum(circles[kPnts,2]))-2, 1) 
        j1 = min(sum(ysC .< maximum(circles[kPnts,2]))+2, length(ysC))  
        loc_XsC = XsC[i0:i1,j0:j1]; loc_YsC = YsC[i0:i1,j0:j1] #局部盒子
        if length(kPnts)==3 # triangle 一个点在三角形内部的充要条件是：它在所有三条边的左侧（对于逆时针顶点顺序）
            @assert get_detTriangle(circles[kPnts,1:2]) > 0  #确保了 Delaunay 三角形的顶点顺序正确性
            isInElem = trues(size(loc_XsC))
            isInElem[isInRight.(loc_XsC,loc_YsC,1,2)] .= false # 如果点在边12的右侧，则标记为false（外部）
            isInElem[isInRight.(loc_XsC,loc_YsC,2,3)] .= false 
            isInElem[isInRight.(loc_XsC,loc_YsC,3,1)] .= false 
            #那么剩下的在三角形每条边左侧
            loc_cellDelaunay = cellDelaunay[i0:i1,j0:j1]
            loc_cellDelaunay[isInElem] .= kE 
            cellDelaunay[i0:i1,j0:j1] = loc_cellDelaunay #分别配好了每个三角形单元编号
        else
            @assert length(kPnts)==4 #收集该单元四个角点的 Voronoi 标记
            @assert get_detTriangle(circles[kPnts[1:3],1:2]) > 0  
            @assert get_detTriangle(circles[kPnts[2:4],1:2]) > 0  
            @assert get_detTriangle(circles[kPnts[[3,4,1]],1:2]) > 0  
            @assert get_detTriangle(circles[kPnts[[4,1,2]],1:2]) > 0 
            #同样，一个点在凸四边形内部的充要条件是：它在所有四条边的左侧 
            isInElem = trues(size(loc_XsC))
            isInElem[isInRight.(loc_XsC,loc_YsC,1,2)] .= false 
            isInElem[isInRight.(loc_XsC,loc_YsC,2,3)] .= false 
            isInElem[isInRight.(loc_XsC,loc_YsC,3,4)] .= false 
            isInElem[isInRight.(loc_XsC,loc_YsC,4,1)] .= false 
            loc_cellDelaunay = cellDelaunay[i0:i1,j0:j1]
            loc_cellDelaunay[isInElem] .= kE 
            cellDelaunay[i0:i1,j0:j1] = loc_cellDelaunay #分配四边形
        end
        doesShow = (kE<10) || (kE<100 && kE%10==0) || (kE<1000 && kE%100==0) || (kE<10000 && kE%1000==0) || kE%10000==0 
        if doesShow 
            println("kE = ", kE) 
            fig_cellDelaunay = Plots.heatmap(cellDelaunay'); display(fig_cellDelaunay) 
        end
            if kE == size(elems,1)
                fig_cellType = Plots.heatmap(cellDelaunay');
                save_plot_with_dir(fig_cellType, "generate_porous_medium/cellDelaunay.png", folder="results")
            end
    end
    # cellDelaunay = cellDelaunay[201:1200,201:1200]
    # fig_cellDelaunay = Plots.heatmap(cellDelaunay'); display(fig_cellDelaunay) 
    #
    # Step 4: to generate the Delaunay diagram together with balls 
    # Step Input: circles[:,1:2] & radius circles[:,3], elems, cellDelaunay 
    # Step Output: cellDelaunay[:,:] (=1,2,... for triangle label 1,2,...; =-1,-2 for balls) 
    #
    for kC = 1:size(circles,1)
        pnt = circles[kC,1:2]; r =  circles[kC,3]
        #只处理圆形可能覆盖的矩形区域，而不是整个网格
        i0 = max(sum(xsC .< (pnt[1] - r))-2, 1) 
        i1 = min(sum(xsC .< (pnt[1] + r))+2, length(xsC))  
        j0 = max(sum(ysC .< (pnt[2] - r))-2, 1) 
        j1 = min(sum(ysC .< (pnt[2] + r))+2, length(ysC))  
        distSq = (XsC[i0:i1,j0:j1] .- pnt[1]).^2 + (YsC[i0:i1,j0:j1] .- pnt[2]).^2
        isNeig = (distSq .< r*r)
        loc_cellDelaunay = cellDelaunay[i0:i1,j0:j1]
        loc_cellDelaunay[isNeig] .= -kC 
        cellDelaunay[i0:i1,j0:j1] = loc_cellDelaunay
        doesShow = (kC<10) || (kC<100 && kC%10==0) || (kC<1000 && kC%100==0) || (kC<10000 && kC%1000==0) || kC%10000==0 
        if doesShow 
            println("with balls: kC = ", kC) 
            fig_cellDelaunay = Plots.heatmap(cellDelaunay'); display(fig_cellDelaunay) 
        end
    end

    #暂时先把标准区域的注释掉
    sta_cellType = cellDelaunay[(2*div(nx_fin,14) + 1): (12*div(nx_fin,14)),(2*div(ny_fin,14) + 1): (12*div(ny_fin,14) )] #
    sta_cellType1 = cellDelaunay[(2*div(nx_fin,14) + 1): (7*div(nx_fin,14)),(2*div(ny_fin,14) + 1): (12*div(ny_fin,14) )] #
    sta_cellType2 = cellDelaunay[(7*div(nx_fin,14) + 1): (12*div(nx_fin,14)),(2*div(ny_fin,14) + 1): (12*div(ny_fin,14) )] #

    sta_XsC = XsC[(2*div(nx_fin,14) + 1): (12*div(nx_fin,14)),(2*div(ny_fin,14) + 1): (12*div(ny_fin,14) )]
    sta_YsC = YsC[(2*div(nx_fin,14) + 1): (12*div(nx_fin,14)),(2*div(ny_fin,14) + 1): (12*div(ny_fin,14))]

    fig_cellType = Plots.heatmap(sta_cellType',size=(400,300)); display(fig_cellType) 
    fig_cellType1 = Plots.heatmap(sta_cellType1',size=(200,300)); display(fig_cellType1)
    fig_cellType2 = Plots.heatmap(sta_cellType2',size=(200,300)); display(fig_cellType2)

    
    # save_plot_with_dir(fig_cellType, "generate_porous_medium_test/fig_cellDelaunay_final.png", folder="results")
    # save_plot_with_dir(fig_cellType1, "generate_porous_medium_test/fig_cellDelaunay_final1.png", folder="results")
    # save_plot_with_dir(fig_cellType2, "generate_porous_medium_test/fig_cellDelaunay_final2.png", folder="results")

    return (exp_XsC=XsC, exp_YsC=YsC, sta_XsC = sta_XsC, sta_YsC = sta_YsC, Num_Delaunay = Num_Delaunay, xs = xs, ys =ys,
        sta_cellType=sta_cellType, cellVoronoi=cellVoronoi, cellDelaunay=cellDelaunay, circles=circles, elems=elems) 
end

function distances(cord_temp,alpha, r_G)
    n = size(cord_temp,1); distance = zeros(3,1); r = zeros(3,1)
    distance[1] = norm(cord_temp[2,:] - cord_temp[1,:]);
    distance[2] = norm(cord_temp[3,:] - cord_temp[2,:]);
    distance[3] = norm(cord_temp[3,:] - cord_temp[1,:]);

    r[1] = ( distance[1] - r_G[2] - r_G[1])/(2*cos(alpha))
    r[2] = ( distance[2] - r_G[3] - r_G[2])/(2*cos(alpha))
    r[3] = ( distance[3] - r_G[1] - r_G[3])/(2*cos(alpha))
    return distance, r
end

#输入: p1 --> p2
function yuanxin(p1, p2, r1, r2, theta)
    
    chord_vector = p2 - p1; d = norm(p2 - p1)  # 弦长
    unit_perp_temp1 = chord_vector / d
    p1_temp = p1 + r1 * unit_perp_temp1; p2_temp = p1 + (d - r2) * unit_perp_temp1

    d_temp = d - r1 -r2;
    chord_vector_temp = p2_temp - p1_temp
    mid_point = (p1_temp + p2_temp) / 2  # 弦中点

    perpendicular = [-chord_vector_temp[2], chord_vector_temp[1]]  # 垂直向量
    unit_perp = perpendicular / norm(perpendicular)
    
    R = d_temp/(2*cos(theta))
    h = R * sin(theta)
    # 两个可能的圆心（在垂直平分线两侧）
    center = zeros(2,2)
    center[1,:] = mid_point + h * unit_perp
    center[2,:] = mid_point - h * unit_perp

    isInRight = (x,y) -> get_detTriangle(vcat(p1',p2',[x y])) < 0 
    println("vcatp = ", vcat(p1',p2'))
    # isInRight.(center[:,1], center[:,2]) = false
    if isInRight(center[1,1], center[1,2]) == true
        center_temp = center[2,:]
    else
        center_temp = center[1,:]
    end

    return center_temp = center_temp
end

function isInbubble(xsC, ysC, XsC, YsC, circles, kPnts, center, R, cellDelaunay)
    #kPnt是单元对应圆索引
        i0 = max(sum(xsC .< minimum(circles[kPnts,1]))-15, 1) #circles储存颗粒中心坐标
        i1 = min(sum(xsC .< maximum(circles[kPnts,1]))+15, length(xsC))  
        j0 = max(sum(ysC .< minimum(circles[kPnts,2]))-15, 1) 
        j1 = min(sum(ysC .< maximum(circles[kPnts,2]))+15, length(ysC)) 

    # cellDelaunay[i0:i1,j0:j1] .= -120

    # fig_domLocation = Plots.heatmap(cellDelaunay'); display(fig_domLocation) 
    # save_plot_with_dir(fig_domLocation, "generate_porous_medium_test/exp_cellType_fin2.png", folder="results")

    sub_domain = [i0 j0; i1 j1]
    loc_XsC = XsC[i0:i1,j0:j1]; loc_YsC = YsC[i0:i1,j0:j1] #局部盒子
    
    isInRight = (x,y,iP,jP) -> get_detTriangle(vcat(circles[kPnts[[iP,jP]],1:2],[x y])) <= 0 #判断是否在右侧
    isInleft  = (x,y,iP,jP) -> get_detTriangle(vcat(circles[kPnts[[iP,jP]],1:2],[x y])) > 0 #判断是否在左侧

    # 预分配：数组的数组（每个元素是矩阵）
    n_pairs = size(kPnts,1)
    # 直接存储矩阵数组（不需要元组）
    distances_mat = Vector{Matrix{Float64}}(undef, n_pairs) #这个时候可以加()
    # distances = Vector{Float64}(undef, n_pairs)
    distances_mat[1] = (loc_XsC .- center[1,1]).^2 + (loc_YsC .- center[1,2]).^2;
    distances_mat[2] = (loc_XsC .- center[2,1]).^2 + (loc_YsC .- center[2,2]).^2;
    distances_mat[3] = (loc_XsC .- center[3,1]).^2 + (loc_YsC .- center[3,2]).^2;

    isNeig = Vector{Matrix{Float64}}(undef, n_pairs)
    # 使用逐元素逻辑与操作 .&
    isNeig[1] = (distances_mat[1] .< R[1]*R[1]) .& isInRight.(loc_XsC,loc_YsC,1,2)
    isNeig[2] = (distances_mat[2] .< R[2]*R[2]) .& isInRight.(loc_XsC,loc_YsC,2,3)
    isNeig[3] = (distances_mat[3] .< R[3]*R[3]) .& isInRight.(loc_XsC,loc_YsC,3,1)

    isInElem = trues(size(loc_XsC))
    isInElem[isInRight.(loc_XsC,loc_YsC,1,2)] .= false # 如果点在边12的右侧，则标记为false（外部）
    isInElem[isInRight.(loc_XsC,loc_YsC,2,3)] .= false 
    isInElem[isInRight.(loc_XsC,loc_YsC,3,1)] .= false 

    loc_cellDelaunay = cellDelaunay[i0:i1,j0:j1]

    for k = 1:size(kPnts,1) #这一点要注意，因为索引数组为kPnts, 并不是丛头开始
        pnt = circles[kPnts[k],1:2]; r =  circles[kPnts[k],3]
        #可以只处理圆形可能覆盖的矩形区域，而不是整个网格
        distSq = (loc_XsC .- pnt[1]).^2 + (loc_YsC .- pnt[2]).^2
        isInElem[distSq .< r*r] .= false 
    end

    isNefin = sum(isNeig) + isInElem
    # println("isInright = ", isNeig[1])
    # loc2 = cellDelaunay[i0:i1,j0:j1]
    # loc2[(distances_mat[1] .< R[1]*R[1]) .& isInRight.(loc_XsC,loc_YsC,1,2)] .= -100-1 #气泡编号
    
    # cellDelaunay[i0:i1,j0:j1] = loc2
    # fig_domLocation = Plots.heatmap(cellDelaunay'); display(fig_domLocation) 
    # save_plot_with_dir(fig_domLocation, "generate_porous_medium_test/exp_cellType_fin5.png", folder="results")
    return (isNefin = isNefin, loc_cellDelaunay = loc_cellDelaunay, sub_domain = sub_domain)
end

function get_qiegedanyuan(cellDelaunay, xs, ys, sta_cellType, Num_Delaunay, circles, elems, nx_fin, ny_fin)
    #暂时先把扩展为1
    elems_temp = elems
    i = div(nx_fin,14); j = div(ny_fin,14)
    hx = xs[2] - xs[1]; hy = ys[2] - ys[1]
    
    nx = 12*i + 2; ny = 10*j + 2
    domSqCore = [-hx -hy; (12*i + 1)*hx (10*j + 1)*hy]  # domain square defined by [a0 b0; a1 b1] 
    xs_new = collect(range(domSqCore[1,1], domSqCore[2,1], length=nx+1))
    ys_new = collect(range(domSqCore[1,2], domSqCore[2,2], length=ny+1))
    xsC = 0.5 * (xs_new[1:end-1] + xs_new[2:end]);  ysC = 0.5 * (ys_new[1:end-1] + ys_new[2:end]) # cell-centers # no need pad here 
    XsC = [x for x in xsC, y in ysC];  YsC = [y for x in xsC, y in ysC] 
    
    # println("domSqCore = ", domSqCore)
    # println("xs = ", xs[2*i]) 外围有一圈鬼点, 但是现在我们事先不考虑鬼点，因为鬼点则是所有边界相当于流体？？？？
    exp_cellType_all  = cellDelaunay[(2*i): (12*i + 1), (2*j): (12*j + 1)] #

    exp_cellType_left = cellDelaunay[(2*i): (7*i), (2*j): (12*j + 1 )] # 
    exp_cellType_right = cellDelaunay[(7*i + 1): (12*i + 1), (2*j): (12*j + 1 )] # 

    # 更简洁的方式
    circles_temp = copy(circles); mask = circles[:, 1] .> xs_new[5*i + 1]
    circles_temp[mask, 1] .= circles_temp[mask, 1] .+ 2*i*hx
    # println("circles_new = ", circles_temp[1,1])
    # println("circles_old", circles[1,1])
# 在循环外部预先定义
    centering = zeros(3, 2)  # 或者根据实际情况初始化

    showDomainLocation = false
    if showDomainLocation
        domLocation = falses(size(exp_cellType_all))
        domLocation[exp_cellType_all .== 11] .= true 
        domLocation[exp_cellType_all .== 55] .= true 
        fig_domLocation = Plots.heatmap(domLocation'); display(fig_domLocation) 
        save_plot_with_dir(fig_domLocation, "generate_porous_medium_test/fsub_quanjuduiying.png", folder="results")
    end

    temp = fill(size(elems,1)+5, 2*i, size(exp_cellType_all,2))
    # println("fill = ", temp)
    exp_cellType_fin = vcat(exp_cellType_left, temp, exp_cellType_right) #补充完空白的

    xuanze = [37,55]; l_temp = zeros(3,2); distance = zeros(3,1); 
    theta = pi/3.0; R = nothing
    for ii = 1:size(xuanze,1)
        kE = xuanze[ii]
        kPnts = elems[kE]; cord_temp = circles_temp[kPnts, 1:2]; cord_old = circles[kPnts,1:2]#返回目前全局单元颗粒坐标

        println("zuobiao = ", kPnts)
        # println("coed_temp", cord_temp)
        # println("coed_temp_old", cord_old)
        r_G =  circles_temp[kPnts,3] #r_G表示固体颗粒半径
        distance, R = distances(cord_temp, theta, r_G); 
        
        temp2 = [1,2,3,1]; center = zeros(size(kPnts,1),2)

        for k = 1:size(kPnts,1)
            p1 = cord_temp[temp2[k],:]; p2 = cord_temp[temp2[k+1],:]
            # println("p1 = ",p1)
            center[k,:] = yuanxin(p1, p2, r_G[temp2[k]], r_G[temp2[k+1]], theta);
            
            # println("centers: = ",center[k,:])

            if ii == 1
                centering[k,:] = center[k,:]
            end
        end
##函数  
        # i0 = max(sum(xsC .< minimum(circles[kPnts,1]))-2, 1) #circles储存颗粒中心坐标
        # i1 = min(sum(xsC .< maximum(circles[kPnts,1]))+2, length(xsC))  
        # j0 = max(sum(ysC .< minimum(circles[kPnts,2]))-2, 1) 
        # j1 = min(sum(ysC .< maximum(circles[kPnts,2]))+2, length(ysC))  

        inInbu = isInbubble(xsC, ysC, XsC, YsC, circles_temp, kPnts, center, R, exp_cellType_fin)

        inInbu.loc_cellDelaunay[inInbu.isNefin .> 0.5] .= -size(elems,1)-ii #气泡编号
        i0 = inInbu.sub_domain[1,1]; j0 = inInbu.sub_domain[1,2]; i1 = inInbu.sub_domain[2,1]; j1 = inInbu.sub_domain[2,2];
        exp_cellType_fin[i0:i1,j0:j1] = inInbu.loc_cellDelaunay

    end

    #表明所有鬼点边界在交点处不一样
    exp_cellType_fin[:,1] .= 120; exp_cellType_fin[end,:] .= 121; exp_cellType_fin[:,end] .= 122; exp_cellType_fin[1,:] .= 123;

    fig_domLocation = Plots.heatmap(exp_cellType_fin'); display(fig_domLocation) 
    save_plot_with_dir(fig_domLocation, "generate_porous_medium_test/exp_cellType_fin.png", folder="results")
    # isInRight = (x,y,iP,jP) -> get_detTriangle(vcat(circles[kPnts[[iP,jP]],1:2],[x y])) < 0


    
    # println("daxiao = ", center[2,1])
    return (exp_cellType_fin = exp_cellType_fin, xsC=xsC, ysC=ysC, XsC=XsC, YsC=YsC, circles_temp=circles_temp, elems_temp = elems_temp, R=R, center1=centering)
end

function index_black!(bound_of_vectors, xsC, ysC, circles_temp, exp_cellType_fin, m1)
#主要需要两个矩阵:一个输入颗粒的值的矩阵，目的返回索引，利用索引去选择glbab矩阵
#boundard_mat: 

    by_vectors = Vector{Vector{Int64}}(undef, size(bound_of_vectors,1)) #记录每一个添加节点的全局节点索引,以边界数进行记录
    println("bound_of_vectors", bound_of_vectors[2])
    cellType_vectors = Vector{Vector{Int64}}(undef, size(bound_of_vectors,1))
    golba = Vector{Vector{Int64}}(undef, size(bound_of_vectors,1))

    cord = zeros(size(by_vectors,1), 2); nx = size(xsC,1); ny = size(ysC,1)
    cord[1,:] = [2, 1]; cord[2,:] = [5*m1+1, 0]; cord[3,:] = [ny-1, 1]; cord[4,:] = [2, 0] #1表示定y 0表示定x
    cord[5,:] = [2, 1]; cord[6,:] = [nx-1, 0]; cord[7,:] = [ny-1, 1]; cord[8,:] = [7*m1+2, 0]
# println("ny = ",ny, 7*m1+2)
    cellType_vectors[1] = Vector{Int64}( vec( exp_cellType_fin[2:(5*div(nx_fin,14))+1, 2] ))
    cellType_vectors[2] = Vector{Int64}( vec( exp_cellType_fin[(5*div(nx_fin,14))+1, 2:end-1] ))
    cellType_vectors[3] = Vector{Int64}(vec( exp_cellType_fin[2:(5*div(nx_fin,14))+1,end-1] ))
    cellType_vectors[4] = Vector{Int64}(vec( exp_cellType_fin[2,2:end-1] ))
    cellType_vectors[5] = Vector{Int64}(vec( exp_cellType_fin[(7*div(nx_fin,14) + 2):end-1, 2] ))
    cellType_vectors[6] = Vector{Int64}(vec( exp_cellType_fin[end-1,2:end-1]))
    cellType_vectors[7] = Vector{Int64}(vec( exp_cellType_fin[(7*div(nx_fin,14)+2) : end-1, end-1] ))
    cellType_vectors[8] = Vector{Int64}(vec( exp_cellType_fin[(7*div(nx_fin,14)+2), 2:end-1] ))

    golba[1] = collect( 2:(5*div(nx_fin,14))+1 )
    golba[2] =  collect( 2:ny-1 )
    golba[3] =  collect( 2:(5*div(nx_fin,14))+1 )
    golba[4] =  collect( 2:ny-1 )
    golba[5] =  collect( (7*div(nx_fin,14) + 2):nx-1 )
    golba[6] =  collect( 2:ny-1 )
    golba[7] =  collect( (7*div(nx_fin,14) + 2):nx-1 )
    golba[8] =  collect( 2:ny-1 )


    Num = size(circles_temp,1)
    for i = 1:size(bound_of_vectors,1)
        by = bound_of_vectors[i][2:end-1]
        by_temp = by[by.<0] #储存的是第i条边界中间固体颗粒
        
        cellType_vectors_temp = cellType_vectors[i]

        by_vectors[i] = Vector{Int64}( vec(zeros(2+size(by_temp,1), 1) )) #储存索引全局值
        for k = 1:length(by_temp)
            keli_value = by_temp[k]
            Index_mat = golba[i][cellType_vectors_temp .== keli_value] #全局索引,从小到大储存索引

            index = div((Index_mat[1] + Index_mat[end]), 2)

            # println("bound_of_vectors 数量: ", length(Index_mat))

            Num = Num + 1
            by_vectors[i][k+1] = Num

            if cord[i,2] == 1
                code_coord = [xsC[index] ysC[Int(cord[i,1])] 0]  
            else
                code_coord = [xsC[Int(cord[i,1])] ysC[index] 0]
            end
            circles_temp = vcat(circles_temp, code_coord)
        end
        
    end
    
    code_node = zeros(8,2)
    code_node[1,:] = [xsC[2], ysC[2]]; code_node[2,:] = [xsC[5*m1+1], ysC[2]]; code_node[3,:] = [xsC[5*m1+1], ysC[end-1]]; code_node[4,:] = [xsC[2], ysC[end-1]];
    code_node[5,:] = [xsC[7*m1+2], ysC[2]]; code_node[6,:] = [xsC[end-1], ysC[2]]; code_node[7,:] = [xsC[end-1], ysC[end-1]]; code_node[8,:] = [xsC[7*m1+2], ysC[end-1]];
    for j = 1:size(bound_of_vectors,1)
        Num = Num + 1
        by_vectors[j][1] = Num; by_vectors[j][end] = Num+1
        if mod(j,4) == 0
            by_vectors[j][end] = Num - 3
        end
        circles_temp = vcat(circles_temp, [code_node[j,1] code_node[j,2] 0])
    end


    return circles_temp, by_vectors
end

function local_index(kPnts, circles, xsC, ysC, min_i, min_j, max_i, max_j, exp_cellType_fin_temp, combin_domain, domain)
        i0 = max(sum(xsC .< minimum(circles[kPnts,1]))-15, min_i) #circles储存颗粒中心坐标
        i1 = min(sum(xsC .< maximum(circles[kPnts,1]))+15, max_i)  
        j0 = max(sum(ysC .< minimum(circles[kPnts,2]))-15, min_j) 
        j1 = min(sum(ysC .< maximum(circles[kPnts,2]))+15, max_j)
        
        loc_exp_cellType_fin_temp = exp_cellType_fin_temp[i0:i1,j0:j1]
        isNeig = in.(loc_exp_cellType_fin_temp, Ref(combin_domain) )

        loc_exp_cellType_fin_temp[isNeig] .= domain
        return i0, i1, j0, j1, loc_exp_cellType_fin_temp
end

function local_index_limit(kPnts, circles, xsC, ysC, min_i, min_j, max_i, max_j, exp_cellType_fin_temp, combin_domain, domain)
        i0 = max(sum(xsC .< minimum(circles[kPnts,1]))-15, min_i) #circles储存颗粒中心坐标
        i1 = min(sum(xsC .< maximum(circles[kPnts,1]))+15, max_i)  
        j0 = max(sum(ysC .< minimum(circles[kPnts,2]))-1, min_j) 
        j1 = min(sum(ysC .< maximum(circles[kPnts,2]))+1, max_j)
        
        loc_exp_cellType_fin_temp = exp_cellType_fin_temp[i0:i1,j0:j1]
        isNeig = in.(loc_exp_cellType_fin_temp, Ref(combin_domain) )

        loc_exp_cellType_fin_temp[isNeig] .= domain
        return i0, i1, j0, j1, loc_exp_cellType_fin_temp
end

function local_matrix_vector(circles_temp)
    #节点坐标全局坐标，我们以此往后写，只是前面的我们不一定索引
    n_pairs_temp = size(circles_temp,1) + 5 #理论上来说应该是局部单元个数，但是现在我们还不知道局部单元有多少个
    local_con_matrix_vector = Vector{Matrix{Int}}(undef, n_pairs_temp)

    # 然后逐个初始化
    for i in 1:n_pairs_temp
        local_con_matrix_vector[i] = zeros(Int, 3, 2)  # 或其他初始化方式
    end
    return local_con_matrix_vector
end

function index_to_local_mat!(kPnts, B, elem_num) #elems_num指的是一个单元有几个通道

    kPnts_temp = vcat(kPnts, kPnts[1])

    idx = (kPnts_temp[1:end-1] .>0) .& (kPnts_temp[2:end] .> 0)
    indices = findall(x -> x > 0, idx)
    A = zeros(Int, size(indices,1), 2)
    for i = 1:size(indices,1)
        A[i,:] = [kPnts_temp[indices[i]], kPnts_temp[indices[i]+1] ]
        B[kPnts_temp[indices[i]], kPnts_temp[indices[i]+1]] = elem_num
    end


    return A, B

end

function sort_and_unique_matrix(matrix)
    # 先按行排序，再删除重复行
    sorted_matrix = sort(matrix, dims=2)
    unique_matrix = unique(sorted_matrix, dims=1)
    return unique_matrix
end

function contruct_gobal_bian_mat(local_con_matrix_vector, elem_list, circles_temp)
    temp_mat = zeros(Int,0,2); N = size(circles_temp, 1)
    A = zeros(Int, N, N)
    for k in elem_list 
        temp_mat = vcat(temp_mat, local_con_matrix_vector[k])
    end

    bianx_quanju_list = sort_and_unique_matrix(temp_mat)

    for i = 1:size(bianx_quanju_list, 1) 
        A[bianx_quanju_list[i,1], bianx_quanju_list[i,2]] = i   #其实就相当于稀疏矩阵, 第i行表示全局边编号， 值表示索引
    end

    A = A + A'
    return A, bianx_quanju_list
end

function fast_local_to_global_mapping!(boundary_inelem_temp, bounda)
    # 创建全局位置的查找字典
    boundary_inelem_temp = sort(boundary_inelem_temp, dims=2)
    bounda = sort(bounda, dims=2)

    global_dict = Dict{Tuple{Int,Int}, Int}()
    
    for j in 1:size(bounda, 1)
        pair = (bounda[j, 1], bounda[j, 2])
        global_dict[pair] = j
    end
    
    # 查找局部边界对应的全局位置
    corresponding_indices = zeros(Int, size(boundary_inelem_temp, 1))
    
    for i in 1:size(boundary_inelem_temp, 1)
        local_pair = (boundary_inelem_temp[i, 1], boundary_inelem_temp[i, 2])
        corresponding_indices[i] = get(global_dict, local_pair, 0)  # 0表示未找到
    end
    
    return corresponding_indices
end

function find_unique_asymmetric_pairs(mat)
    n = size(mat, 1)
    asymmetric_pairs = Tuple{Int, Int}[]
    
    # 只检查上三角部分，避免重复计数 (i,j) 和 (j,i)
    for i in 1:n
        for j in (i+1):n
            if (mat[i, j] == 0 && mat[j, i] != 0) || (mat[i, j] != 0 && mat[j, i] == 0)
                push!(asymmetric_pairs, (i, j))
            end
        end
    end
    
    Nb = length(asymmetric_pairs)
    boundary_index_temp = zeros(Int, Nb, 2)
    
    for (idx, (i, j)) in enumerate(asymmetric_pairs)
        boundary_index_temp[idx, 1] = i
        boundary_index_temp[idx, 2] = j
    end
    
    println("找到 ", Nb, " 个唯一的不对称边对")

    boundary_mat = zeros(size(mat))
    boundary_mat[4,9] = 1; boundary_mat[4,72]=1; boundary_mat[5,68]=1; boundary_mat[5,69]=1; boundary_mat[9,73]=1; boundary_mat[69,72]=1
    boundary_mat[46,73]=2; boundary_mat[46,76]=2; boundary_mat[7,74]=2; boundary_mat[7,76]=2
    boundary_mat[26,56]=3; boundary_mat[26,71]=3; boundary_mat[47,48]=3; boundary_mat[47,70]=3; boundary_mat[48,74]=3; boundary_mat[56,70]=3
    boundary_mat[2,58]=4; boundary_mat[2,68]=4; boundary_mat[59,71]=4; boundary_mat[58,59]=4

    boundary_mat = boundary_mat + boundary_mat'

    return boundary_index_temp, boundary_mat, Nb
end

function Information_Matrix!(exp_cellType_fin, xsC, ysC, XsC, YsC, circles_temp0, nx_fin, ny_fin, elems, R, center)
    #exp_cellType_fin: 单元全局分布
    #circles_temp: 储存颗粒中心坐标和半径
    # println("ysC_N = ", max(ysC))
    #Step 1: 处理好边界项，需要重新分配单元的边界以及相关数据
    #中左

    #重新分配左侧域边表示 如第2条边应该是 3 --> 5指的是颗粒点3 --> 5
    n1 = 8; m1 = div(nx_fin,14); elems_temp = copy(elems); N_cir = size(circles_temp0,1)
    bound_of_vectors = Vector{Vector{Float64}}(undef, n1)
    bound_of_vectors[1] = unique(exp_cellType_fin[2:(5*div(nx_fin,14))+1, 2])
    bound_of_vectors[2] = unique(exp_cellType_fin[(5*div(nx_fin,14))+1, 2:end-1])
    bound_of_vectors[3] = unique(exp_cellType_fin[2:(5*div(nx_fin,14))+1,end-1])
    bound_of_vectors[4] = unique(exp_cellType_fin[2,2:end-1])
    bound_of_vectors[5] = unique(exp_cellType_fin[(7*div(nx_fin,14) + 2):end-1, 2]) #计算中间剖分域编号
    bound_of_vectors[6] = unique(exp_cellType_fin[end-1,2:end-1])
    bound_of_vectors[7] = unique(exp_cellType_fin[(7*div(nx_fin,14)+2) : end-1, end-1])
    bound_of_vectors[8] = unique(exp_cellType_fin[(7*div(nx_fin,14)+2), 2:end-1])

    
    circles_temp, by_vectors = index_black!(bound_of_vectors, xsC, ysC, circles_temp0, exp_cellType_fin, m1) #circles_temp大小要包含所有单元顶点，即圆心编号

    bianx_in_elems_mat = zeros(Int, 77, 77) #该矩阵储存的是链接边，如A(1,2) = 1, 那么1,2便是一个未知量
    # 绘制散点图
    fig = scatter(vec(circles_temp[:,1]), vec(circles_temp[:,2]), label="sandian")
    save_plot_with_dir(fig, "generate_porous_medium_test/fig_scatter.png", folder="results")
    # println("circles_temp: ", circles_temp)
    println("by_vectors: ", by_vectors)

    local_con_matrix_vector = local_matrix_vector(circles_temp) #local_con_matrix_vector大小与单元数量相关， 储存每一个单元信息

    exp_cellType_fin_temp = copy(exp_cellType_fin)
    i0, i1, j0, j1, loc_exp_cellType_fin_temp = local_index([15,39,5], circles_temp, xsC, ysC, 1, 1, 5*m1 + 1, length(ysC), exp_cellType_fin_temp, [3, 4, 11], 3)

    #第五条边
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp
    elems_temp[3] = [15, 39, 5, by_vectors[2][1]]; kPnts = elems_temp[3];
    local_con_matrix_vector[3], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 3)

    elems_temp[29] = [22, 15, by_vectors[2][3]]; kPnts = [22, 15, by_vectors[2][3], 0]; element_index = [3, 29]
    local_con_matrix_vector[29], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 29)

    elems_temp[39] = [30, by_vectors[2][3], by_vectors[2][4], 40]; kPnts = [30,0,by_vectors[2][3],by_vectors[2][4],0,40];
    local_con_matrix_vector[39], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 39)
    element_index = vcat(element_index, 39)
    #此时39比较还是之前的域，有重复的
    i0, i1, j0, j1, loc_exp_cellType_fin_temp = local_index(elems_temp[39], circles_temp, xsC, ysC, 1, 1, 5*m1 + 1, length(ysC), exp_cellType_fin_temp, [39, 41, 47], 39)
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp
    
    elems_temp[57] = [21, by_vectors[2][4], by_vectors[2][5]]; kPnts = [21, 0, by_vectors[2][4], by_vectors[2][5]]
    local_con_matrix_vector[57], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 57)
    element_index = vcat(element_index, 57)

    #第三条
    elems_temp[65] = [21, by_vectors[3][1], by_vectors[3][3]]; kPnts = [21, by_vectors[3][1], by_vectors[3][3]]
    local_con_matrix_vector[65], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 65)
    element_index = vcat(element_index, 65)
    i0, i1, j0, j1, loc_exp_cellType_fin_temp = local_index(elems_temp[65], circles_temp, xsC, ysC, 1, 1, 5*m1 + 1, length(ysC), exp_cellType_fin_temp, [65, 70], 65)
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp

    elems_temp[64] = [26, 21, by_vectors[3][3]]; kPnts = [26, 21, by_vectors[3][3]]
    local_con_matrix_vector[64], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 64)
    element_index = vcat(element_index, 64)

    elems_temp[60] = [31, 26, by_vectors[3][4], by_vectors[4][4]]; kPnts = [31, 26, by_vectors[3][4], by_vectors[4][4], 0]
    local_con_matrix_vector[60], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 60)
    element_index = vcat(element_index, 60)
    println("mat = ", local_con_matrix_vector[60])
    i0, i1, j0, j1, loc_exp_cellType_fin_temp = local_index(elems_temp[60], circles_temp, xsC, ysC, 1, 1, 5*m1 + 1, length(ysC), exp_cellType_fin_temp, [60, 63], 60)
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp

    #第四条
    elems_temp[44] = [23, 31, by_vectors[4][4], by_vectors[4][3]]; kPnts = [23, 31, 0, by_vectors[4][4], by_vectors[4][3],0]
    local_con_matrix_vector[44], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 44)
    element_index = vcat(element_index, 44)

    elems_temp[34] = [2, 23, by_vectors[4][3]]; kPnts = [2, 23, 0, by_vectors[4][3]]
    local_con_matrix_vector[34], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat,34)
    element_index = vcat(element_index, 34)

    elems_temp[14] = [by_vectors[4][5], 5, 2]; kPnts = [by_vectors[4][5], 5, 2]
    local_con_matrix_vector[14], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 14)
    element_index = vcat(element_index, 14)
    # println("mat = ", local_con_matrix_vector[60])
    i0, i1, j0, j1, loc_exp_cellType_fin_temp = local_index(elems_temp[14], circles_temp, xsC, ysC, 1, 1, 5*m1 + 1, length(ysC), exp_cellType_fin_temp, [14, 5], 14)
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp

    #第五条边
    elems_temp[11] = [by_vectors[5][1], 4, by_vectors[8][2]]; kPnts = [by_vectors[5][1], 4, by_vectors[8][2]]
    local_con_matrix_vector[11], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 11)
    element_index = vcat(element_index, 11)

    #15号引起的改变
    elems_temp[12] = [4, 29, by_vectors[8][2]]; kPnts = [4, 29, by_vectors[8][2]]
    local_con_matrix_vector[12], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 12)
    element_index = vcat(element_index, 12)

    elems_temp[20] = [by_vectors[8][2],29,43]; kPnts = [by_vectors[8][2],29,43]
    local_con_matrix_vector[20], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 20)
    element_index = vcat(element_index, 20)

    elems_temp[26] = [by_vectors[8][2], 43, 42]; kPnts = [by_vectors[8][2], 43, 42]
    local_con_matrix_vector[26], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 26)
    element_index = vcat(element_index,26)

    kPnts = [4, 9, 29]
    local_con_matrix_vector[9], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 9)
    element_index = vcat(element_index, 9)

    elems_temp[6] = [9, by_vectors[5][4], 46]; kPnts = [9, by_vectors[5][4], 46]
    local_con_matrix_vector[6], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 6)
    element_index = vcat(element_index, 6)

    #情况特殊, 额外添加一点, 在by_vectors中
    x0y0 = circles_temp[19,1:2]; by_vectors[6] = vcat(by_vectors[6][1:2], size(circles_temp,1)+1, by_vectors[6][3:end])
    circles_temp = vcat(circles_temp, [xsC[end-1] x0y0[2] 0])

    elems_temp[25] = [38, 46, by_vectors[6][3], 19]; kPnts = [38, 46, by_vectors[6][3], 19]
    local_con_matrix_vector[25], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 25)
    element_index = vcat(element_index, 25)
    # println("mat = ", local_con_matrix_vector[60])
    i0, i1, j0, j1, loc_exp_cellType_fin_temp = local_index(elems_temp[25], circles_temp, xsC, ysC, 7*m1 + 1, 1, length(xsC), length(ysC), exp_cellType_fin_temp, [24, 25, 30], 25)
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp

    elems_temp[46] = [50, 19, by_vectors[6][3], 7]; kPnts = [50, 19, by_vectors[6][3], 7]
    local_con_matrix_vector[46], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 46)
    element_index = vcat(element_index, 46)
    i0, i1, j0, j1, loc_exp_cellType_fin_temp = local_index(elems_temp[46], circles_temp, xsC, ysC, 7*m1 + 1, 1, length(xsC), length(ysC), exp_cellType_fin_temp, [43, 46, 50], 46)
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp

    elems_temp[67] = [7, by_vectors[6][5], 48]; kPnts = [7, by_vectors[6][5], 48]
    local_con_matrix_vector[67], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 67)
    element_index = vcat(element_index, 67)

    #第七条
    elems_temp[66] = [3, 48, 47]; kPnts = [3, 48, 47]
    local_con_matrix_vector[66], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 66)
    element_index = vcat(element_index, 66)
    i0, i1, j0, j1, loc_exp_cellType_fin_temp = local_index(elems_temp[66], circles_temp, xsC, ysC, 7*m1 + 1, 1, length(xsC), length(ysC), exp_cellType_fin_temp, [66, 71], 66)
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp

    #第八条
    elems_temp[47] = [47, by_vectors[8][4], 28]; kPnts = [47, by_vectors[8][4], 0, 28]
    local_con_matrix_vector[47], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 47)
    element_index = vcat(element_index, 47)
    i0, i1, j0, j1, loc_exp_cellType_fin_temp = local_index_limit(elems_temp[47], circles_temp, xsC, ysC, 7*m1 + 1, 1, length(xsC), length(ysC), exp_cellType_fin_temp, [55, 57], 47)
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp


    #情况特殊, 额外添加一点, 在by_vectors中
    x0y0 = circles_temp[27,1:2]; by_vectors[8] = vcat(by_vectors[8][1:3], size(circles_temp,1)+1, by_vectors[8][4:end])
    circles_temp = vcat(circles_temp, [xsC[7*m1+2] x0y0[2] 0])

    elems_temp[55] = [by_vectors[8][5], by_vectors[8][4], 27, 28]; kPnts = [by_vectors[8][5], by_vectors[8][4], 27, 28, 0]
    local_con_matrix_vector[55], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 55)
    element_index = vcat(element_index, 55)
    i0, i1, j0, j1, loc_exp_cellType_fin_temp = local_index_limit(elems_temp[55], circles_temp, xsC, ysC, 7*m1 + 1, 1, length(xsC), length(ysC), exp_cellType_fin_temp, [55, 47, 41], 55)
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp
    # println("i0, i1, j0, j1 = ", j0)


    showDomainLocation1 = false
    if showDomainLocation1
        exp_cellType_fin_temp[i0:i1,j0:j1] .= -10
        fig_domLocation = Plots.heatmap(exp_cellType_fin_temp'); display(fig_domLocation) 
        save_plot_with_dir(fig_domLocation, "generate_porous_medium_test/showDomainjuxing.png", folder="results")
    end

    #以后有错误记得排查，感觉这一点有问题
    elems_temp[41] = [by_vectors[8][4], by_vectors[8][3], 42, 27]; kPnts = [by_vectors[8][4], by_vectors[8][3], 0, 42, 27]
    local_con_matrix_vector[41], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 41)
    element_index = vcat(element_index, 41)
    i0, i1, j0, j1, loc_exp_cellType_fin_temp = local_index_limit(elems_temp[41], circles_temp, xsC, ysC, 7*m1 + 1, 1, length(xsC), length(ysC), exp_cellType_fin_temp, [41,39,37], 41)
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp
    # exp_cellType_fin_temp[i0:i1,j0:j1] .= -10

    elems_temp[50] = [by_vectors[8][3], by_vectors[8][2], 42]; kPnts = [by_vectors[8][3], by_vectors[8][2], 42,0]
    local_con_matrix_vector[50], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 50)
    element_index = vcat(element_index, 50)
    i0, i1, j0, j1, loc_exp_cellType_fin_temp = local_index_limit(elems_temp[50], circles_temp, xsC, ysC, 1, 1, length(xsC), length(ysC), exp_cellType_fin_temp, [29,37], 50)
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp
    
    
    elems_temp[49] = [30, 40, by_vectors[2][4], 21]; kPnts = [30, 40, 0, by_vectors[2][4], 0, 21]
    local_con_matrix_vector[49], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 49)
    element_index = vcat(element_index, 49)
    i0, i1, j0, j1, loc_exp_cellType_fin_temp = local_index_limit(elems_temp[49], circles_temp, xsC, ysC, 1, 1, length(xsC), length(ysC), exp_cellType_fin_temp, 49, 49)
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp

    elems_temp[35] = [24, 22, by_vectors[2][4], 30]; kPnts = [24, 22, 0, by_vectors[2][4], 0, 30]
    local_con_matrix_vector[35], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, 35)
    element_index = vcat(element_index, 35)
    i0, i1, j0, j1, loc_exp_cellType_fin_temp = local_index_limit(elems_temp[35], circles_temp, xsC, ysC, 1, 1, length(xsC), length(ysC), exp_cellType_fin_temp, 35, 35)
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp
    println("local_con_matrix_vector = ",local_con_matrix_vector[35])
    #中间处理
    # 扩张elem_temp, 这里是为了扩张，因为一开始大小已经定了下来
    circles_temp[by_vectors[8][2],2] = circles_temp[15, 2]
    n_elems_temp = size(elems_temp,1)
    append!(elems_temp, [
        [by_vectors[2][1], by_vectors[8][6], by_vectors[8][2], 15],
        [15, by_vectors[8][2], by_vectors[8][3], by_vectors[2][3]],
        [by_vectors[2][3], by_vectors[8][3], by_vectors[8][4]],
        [by_vectors[2][3], by_vectors[8][4], by_vectors[8][5], by_vectors[2][4]],
        [by_vectors[2][4], by_vectors[8][5], 47, by_vectors[2][5]]
    ])
    kPnts = [by_vectors[2][1], by_vectors[8][6], by_vectors[8][2], 15]
    local_con_matrix_vector[n_elems_temp + 1], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat,n_elems_temp + 1)
    element_index = vcat(element_index, n_elems_temp + 1)
    i0, i1, j0, j1, loc_exp_cellType_fin_temp = local_index_limit(elems_temp[n_elems_temp + 1], circles_temp, xsC, ysC, 1, 1, length(xsC), length(ysC), exp_cellType_fin_temp, 78, n_elems_temp +1)
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp

    kPnts = [15, by_vectors[8][2], by_vectors[8][3], 0, by_vectors[2][3]]
    local_con_matrix_vector[n_elems_temp + 2], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat,n_elems_temp + 2)
    element_index = vcat(element_index, n_elems_temp + 2)
    i0, i1, j0, j1, loc_exp_cellType_fin_temp = local_index_limit(elems_temp[n_elems_temp + 2], circles_temp, xsC, ysC, 1, 1, length(xsC), length(ysC), exp_cellType_fin_temp, 78, n_elems_temp +2)
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp

    #空白弯曲点
    y_jiaodian = sqrt(R[2]^2 - (xsC[5*m1+1] - center[2,1])^2) + center[2,2]
    coord_xy = [xsC[5*m1+1], y_jiaodian]; kPnts = [by_vectors[2][3], by_vectors[8][3], by_vectors[8][4]]
    circles_temp[by_vectors[2][3],1:2] = coord_xy
    isInleft = (x,y,iP,jP) -> get_detTriangle(vcat(circles_temp[kPnts[[iP,jP]],1:2],[x y])) > 0 #判断是否在左侧
    i0 = max(sum(xsC .< minimum(circles_temp[kPnts,1]))-2, 1) #circles储存颗粒中心坐标
    i1 = min(sum(xsC .< maximum(circles_temp[kPnts,1]))+2, length(xsC))  
    j0 = max(sum(ysC .< minimum(circles_temp[kPnts,2]))-2, 1) 
    j1 = min(sum(ysC .< maximum(circles_temp[kPnts,2]))+2, length(ysC)) 
    loc_XsC = XsC[i0:i1,j0:j1]; loc_YsC = YsC[i0:i1,j0:j1] #局部盒子
    loc_exp_cellType_fin_temp = exp_cellType_fin_temp[i0:i1,j0:j1]
    loc_exp_cellType_fin_temp[isInleft.(loc_XsC,loc_YsC,3,1) .& (loc_exp_cellType_fin_temp .== 78)] .= n_elems_temp + 3
    element_index = vcat(element_index, n_elems_temp + 3)
    kPnts = [by_vectors[2][3], 0, by_vectors[8][3], by_vectors[8][4]]
    local_con_matrix_vector[n_elems_temp + 3], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat,n_elems_temp + 3)
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp

    kPnts = [by_vectors[2][3], by_vectors[8][4], by_vectors[8][5], by_vectors[2][4]]
    i0 = max(sum(xsC .< minimum(circles_temp[kPnts,1]))-2, 1) #circles储存颗粒中心坐标
    i1 = min(sum(xsC .< maximum(circles_temp[kPnts,1]))+2, length(xsC))  
    j0 = max(sum(ysC .< minimum(circles_temp[kPnts,2]))-2, 1) 
    j1 = min(sum(ysC .< maximum(circles_temp[kPnts,2]))+2, length(ysC)) 
    loc_XsC = XsC[i0:i1,j0:j1]; loc_YsC = YsC[i0:i1,j0:j1] #局部盒子
    loc_exp_cellType_fin_temp = exp_cellType_fin_temp[i0:i1,j0:j1]
    loc_exp_cellType_fin_temp[isInleft.(loc_XsC,loc_YsC,1,2) .& (loc_exp_cellType_fin_temp .== 78)] .= n_elems_temp + 4
    element_index = vcat(element_index, n_elems_temp + 4)
    kPnts = [by_vectors[2][3], by_vectors[8][4], by_vectors[8][5], 0, by_vectors[2][4]]
    local_con_matrix_vector[n_elems_temp + 4], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat,n_elems_temp + 4)
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp
    #

    kPnts = [by_vectors[2][4], 0, by_vectors[8][5], 47, by_vectors[2][5]]
    local_con_matrix_vector[n_elems_temp + 5], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat,n_elems_temp + 5)
    element_index = vcat(element_index, n_elems_temp + 5)
    i0, i1, j0, j1, loc_exp_cellType_fin_temp = local_index_limit(elems_temp[n_elems_temp + 5], circles_temp, xsC, ysC, 1, 1, length(xsC), length(ysC), exp_cellType_fin_temp, 78, n_elems_temp +5)
    exp_cellType_fin_temp[i0:i1,j0:j1] = loc_exp_cellType_fin_temp  #78个
    # cellDelaunay[i0:i1,j0:j1] .= -120

    #补充，即使没有被切割的只要修改了编号连通的都需要修改

    #剩下的索引全是满引，指的是通道数==颗粒数
    exp_cellType_fin_temp_neibu = exp_cellType_fin_temp[2:end-1,2:end-1]
    elem_list = unique(exp_cellType_fin_temp_neibu[exp_cellType_fin_temp_neibu .> 0]) #当前区域所有单元编号，代号

    for k = 1:size(elem_list, 1) 
        j = elem_list[k] #标识编号j
        if j in element_index
            println("$j 在数组中")
        else
            kPnts = elems[j] #返回当前元的编号, 因为这是老的所以没有动
            # println("kpnts = ", j, "  ", kPnts)
            local_con_matrix_vector[j], bianx_in_elems_mat = index_to_local_mat!(kPnts, bianx_in_elems_mat, j)
        end
    end


    # 综上: 已知elem_list 当前区域所有单元编号,  local_con_matrix_vector储存局部边界连通信息, bianx_in_elems_mat 储存全局局部边界连通信息, 不等于表示连通，值表示所属单元
    #下面开始边界全局编号, 同样使用矩阵形式, 
    bianx_gobal_mat, bianx_quanju_list = contruct_gobal_bian_mat(local_con_matrix_vector, elem_list, circles_temp)

    boundary_inelem_temp, boundary_index_temp, N = find_unique_asymmetric_pairs(bianx_in_elems_mat)

    for i = 1:N
        println("bianx_gobal_mat_in_elem = ", bianx_in_elems_mat[boundary_inelem_temp[i,1], boundary_inelem_temp[i,2]] + bianx_in_elems_mat[boundary_inelem_temp[i,2], boundary_inelem_temp[i,1]])
        
    end

    boundary_global_indices = fast_local_to_global_mapping!(boundary_inelem_temp, bianx_quanju_list) #返回边界全局索引

    println("bianx_gobal_mat = ", bianx_gobal_mat[by_vectors[2][1], by_vectors[8][6]], "  ", bianx_gobal_mat[by_vectors[8][6], by_vectors[2][1]])#bianx_in_elems_mat[46,by_vectors[6][3]], "   ", bianx_in_elems_mat[by_vectors[6][3], 46],"   ", bianx_in_elems_mat[47, by_vectors[8][5]])
    println("boundary_inelem_temp = ", boundary_inelem_temp)
    println("bianx_quanju_list = ", bianx_quanju_list)
    println("boundary_global_indices = ", boundary_global_indices)
    # fig_domLocation = Plots.heatmap(cellDelaunay'); display(fig_domLocation) 
    # save_plot_with_dir(fig_domLocation, "generate_porous_medium_test/exp_cellType_fin2.png", folder="results")

    println("danyuan = ", y_jiaodian)
    println("shuzhi_bianhao = ", bound_of_vectors)
    println("danyuan = ", by_vectors)

    println("danyuan1 = ", elems_temp[34])
    println("danyuan2 = ", elems_temp[44])
    println("danyuan2 = ", elems_temp[12])

    showDomainLocation = false
    if showDomainLocation
        domLocation = falses(size(exp_cellType_fin_temp))
        domLocation[(exp_cellType_fin_temp .==  34)] .= true 
        # domLocation[exp_cellType_fin .== 11] .= true 
        fig_domLocation = Plots.heatmap(domLocation'); display(fig_domLocation) 
        save_plot_with_dir(fig_domLocation, "generate_porous_medium_test/showDomainLocation.png", folder="results")
    end


    println("num_size = ", size(circles_temp,1))
    # bound_of_vect = unique(exp_cellType_fin_temp[2:(5*div(nx_fin,14))+1, 160])
    println("boundary_inelem_temp = ", boundary_inelem_temp)


    println("bound_all_num = ", size(bianx_quanju_list))
    #矩阵k与elem(i)的i的关系，vector_elem
    # vector_elem = zeros(Int, size(circles_temp,1), 2)

    return (boundary_global_indices = boundary_global_indices, boundary_index_temp = boundary_index_temp, boundary_inelem_temp = boundary_inelem_temp, bianx_in_elems_mat = bianx_in_elems_mat, bianx_gobal_mat = bianx_gobal_mat, 
            elem_list = elem_list, local_con_matrix_vector = local_con_matrix_vector, bianx_quanju_list = bianx_quanju_list, elems_temp = elems_temp, circles_temp = circles_temp,
            exp_cellType_fin_temp=exp_cellType_fin_temp, XsC=XsC, YsC=YsC)
end
#单元一定要有顺序排列，逆时针排序
#boundary_global_indices: 储存边界全局边索引
#boundary_index_temp: 储存边界索引类型(边界编号1,2,3,4)
#boundary_inelem_temp: 储存边界i-->j信息
#bianx_in_elems_mat:i,j, value 表示value = 0 ==> i-->j不连通, 否则返回i--j所属单元 非对称阵
#bianx_gobal_mat: i, j, value   i --> j 第value条边, 后续组装刚度矩阵使用 对称阵
#bianx_quanju_list: 所有边的索引
#elem_list: 储存所有单元
#local_con_matrix_vector: 向量矩阵: k, Mat 第k单元边的信息, Mat储存局部单元链接(i,j): i --> j
#bianx_quanju_list: N,2  共N条边, 值表示i --> j
#elems_temp： 向量数组: 储存单元顶点

#--------------------------------------------------------子区间求解-----------------------------
function get_porousMedium_subDomain(XsC, YsC, cellType, local_con_matrix_vector, bianx_in_elems_mat, boundary_index_temp, kDomain)
    #celltype XsC YsC 可以表示标准计算区域
    @assert in(kDomain, unique(cellType)) # 确保kDomain确实存在
    Index = local_con_matrix_vector[kDomain]
    #首先判断有几个边界条件
    num = 0
    for k = 1: size(Index,1) 
        i = Index[k,1]; j = Index[k,2]
        num += (bianx_in_elems_mat[i,j] == 0 && bianx_in_elems_mat[i,j] != 0) || (bianx_in_elems_mat[i,j] != 0 && bianx_in_elems_mat[i,j] == 0)
    end
    
    demo_subdomain = zeros(Int,2,2)
    Is = [i for i in 1:size(cellType,1), j in 1:size(cellType,2)]
    Js = [j for i in 1:size(cellType,1), j in 1:size(cellType,2)] 
    iMin = max(minimum(Is[cellType.==kDomain])-1, 1); 
    iMax = min(maximum(Is[cellType.==kDomain])+1, size(cellType,1));
    jMin = max(minimum(Js[cellType.==kDomain])-1, 1); 
    jMax = min(maximum(Js[cellType.==kDomain])+1, size(cellType,2));
    demo_subdomain[1,1] = iMin; demo_subdomain[1,2] = jMin; demo_subdomain[2,1] = iMax; demo_subdomain[2,2] = jMax
    sbCellType = cellType[iMin:iMax, jMin:jMax]
    # change glb labels to subdomain local labels 
    posUnique = unique(sbCellType[(sbCellType .> 0) .& (sbCellType .!== kDomain)]) #有几个其他三角形的
    # 找出子区域内的所有负值（固体颗粒）
    negUnique = unique(sbCellType[sbCellType .< 0])
    println("posUnique = ", posUnique, ", negUnique = ", negUnique) 
    locCellType = -1*ones(Int, size(sbCellType)) #这里做了一些修改
    locCellType[sbCellType .== kDomain] .= 0
    for iP = 1:size(Index, 1) ## iP in eachindex(posUnique):  linear indexing
        i = Index[iP,1]; j = Index[iP,2] #隔壁单元赋为边界条件，进行处理
        if boundary_index_temp[i,j] == 0
            locCellType[sbCellType .== bianx_in_elems_mat[j,i]] .= iP
        elseif boundary_index_temp[i,j] == 1
            locCellType[sbCellType .== 120] .= iP
        elseif boundary_index_temp[i,j] == 2
            locCellType[sbCellType .== 121] .= iP
        elseif boundary_index_temp[i,j] == 3
            locCellType[sbCellType .== 122] .= iP
        else 
            locCellType[sbCellType .== 123] .= iP
        end
    end 
    for iN in eachindex(negUnique)
        locCellType[sbCellType .== negUnique[iN]] .= -iN #固体区域重新编写
    end 
    showDomainLocation = false
    if showDomainLocation
        domLocation = falses(size(cellType))
        domLocation[cellType .== kDomain] .= true 
        fig_cellType = Plots.heatmap(cellType'); display(fig_cellType) 
        fig_domLocation = Plots.heatmap(domLocation'); display(fig_domLocation) 
        save_plot_with_dir(fig_domLocation, "generate_porous_medium_test/fsub_quanjuduiying.png", folder="results")
    end
    # fig_locCellType = Plots.heatmap(locCellType'); display(fig_locCellType) 
    # save_plot_with_dir(fig_locCellType, "generate_porous_medium_test/fsub_cellDelaunay_20.png", folder="results")
    return (sbXsC=XsC[iMin:iMax, jMin:jMax], sbYsC=YsC[iMin:iMax, jMin:jMax], 
            sbCellType=sbCellType, locCellType=locCellType,demo_subdomain=demo_subdomain)
end
# res = generate_porous_medium(1400,1400) 

function get_StokesFlow_matrixFlowOut(cellType, xsC, ysC, local_con_matrix_vector, kDomain) 
    numInOutLets = size(local_con_matrix_vector[kDomain], 1) #这里做了修改
    flows = Vector(undef, numInOutLets)
    for iPresBdry = 1:numInOutLets 
        flows[iPresBdry] = solve_StokesFlow(cellType, xsC, ysC, iPresBdry) 
    end
    matrixFlowOut = hcat([flows[i].flowOutRates for i=1:numInOutLets]...) 
    println("\n matrixFlowOut = ", matrixFlowOut, "\n")
    return matrixFlowOut
end
# res = generate_porous_medium(1400,1400) 
# sbRes = get_porousMedium_subDomain(res.XsC, res.YsC, res.cellType, 30) 
# mat = get_StokesFlow_matrixFlowOut(sbRes.locCellType, sbRes.sbXsC[:,1], sbRes.sbYsC[1,:]) 

# nx_fin = 560; ny_fin = 560
# res = generate_porous_medium(560,560,400, 1.1) #560 560
#    # #shuzhi_bianhao = [11, -15, 29, 37, 39, 41, 47, 55, 57, -47]
# qiege = get_qiegedanyuan(res.cellDelaunay, res.xs, res.ys, res.sta_cellType, res.Num_Delaunay, res.circles, res.elems, 560,560)
# infor = Information_Matrix!(qiege.exp_cellType_fin, qiege.xsC, qiege.ysC, qiege.XsC, qiege.YsC, qiege.circles_temp, nx_fin, ny_fin, qiege.elems_temp, qiege.R, qiege.center1)
# sbRes = get_porousMedium_subDomain(infor.XsC, infor.YsC, infor.exp_cellType_fin_temp, infor.local_con_matrix_vector, infor.bianx_in_elems_mat, infor.boundary_index_temp,49)
# mat = get_StokesFlow_matrixFlowOut(sbRes.locCellType, sbRes.sbXsC[:,1], sbRes.sbYsC[1,:], infor.local_con_matrix_vector, 49) 

# 根据elem锁定局部区域，计算局部区域传导率
function get_all_porousMedium_subDomain(XsC, YsC, exp_cellType_fin_temp, local_con_matrix_vector, bianx_in_elems_mat, boundary_index_temp, elem_list)
    N = size(local_con_matrix_vector,1)
    hydraulic_conductance = Vector{Matrix{Float64}}(undef, N)
    # 然后逐个初始化
    for i = 1:N
        hydraulic_conductance[i] = zeros(3, 3)  # 或其他初始化方式
    end

    for kDomain in elem_list
        println("kdomain = ", kDomain)
        sbRes = get_porousMedium_subDomain(XsC, YsC, exp_cellType_fin_temp, local_con_matrix_vector, bianx_in_elems_mat, boundary_index_temp, kDomain)
        hydraulic_conductance[kDomain] = get_StokesFlow_matrixFlowOut(sbRes.locCellType, sbRes.sbXsC[:,1], sbRes.sbYsC[1,:], local_con_matrix_vector, kDomain) 
    end

    return hydraulic_conductance
end

# nx_fin = 560; ny_fin = 560
# res = generate_porous_medium(nx_fin,ny_fin,400, 1.1) #560 560
# qiege = get_qiegedanyuan(res.cellDelaunay, res.xs, res.ys, res.sta_cellType, res.Num_Delaunay, res.circles, res.elems, nx_fin,ny_fin)
# infor = Information_Matrix!(qiege.exp_cellType_fin, qiege.xsC, qiege.ysC, qiege.XsC, qiege.YsC, qiege.circles_temp, nx_fin, ny_fin, qiege.elems_temp, qiege.R, qiege.center1)
# # hydraulic_conductance = get_all_porousMedium_subDomain(infor.XsC, infor.YsC, infor.exp_cellType_fin_temp, infor.local_con_matrix_vector, infor.bianx_in_elems_mat, infor.boundary_index_temp, infor.elem_list)

# 找出在 matrixA 中但不在 matrixB 中的行
function matrix_setdiff(A::Matrix, B::Matrix)
    # 将每行转换为元组以便比较
    setA = Set([tuple(row...) for row in eachrow(A)])
    setB = Set([tuple(row...) for row in eachrow(B)])
    
    # 计算差集
    diff_set = setdiff(setA, setB)
    
    # 转换回矩阵
    if isempty(diff_set)
        return zeros(0, 2)
    else
        return reduce(vcat, [reshape(collect(pair), 1, 2) for pair in diff_set])
    end
end

function dsolve_pressure(hydraulic_conductance, boundary_global_indices, boundary_index_temp, boundary_inelem_temp, bianx_in_elems_mat, local_con_matrix_vector, bianx_quanju_list)
#hydraulic_conductance: A 是一个类似 local_con_matrix_vector 的向量矩阵， Mat = A[k];  k表示第几个单元的导水率
#边界也求: step1: 先组装控制方程，再处理边界条件
#边界不求: 先组装控制方程，处理控制方程边界条件;    先给出边界条件，再组装控制方程

    paraSrc = (x,y) -> 0.0
    paraboundary1 = (x,y) -> 0.0
    paraboundary2 = (x,y) -> 0.0
    paraboundary3 = (x,y) -> 0.0
    paraboundary4 = (x,y) -> 1.0

    Np = size(bianx_quanju_list, 1); Np_boundary = size(boundary_inelem_temp, 1)

    coords = reshape(collect(1:Np), (Np, 1))

    x_coords = coords  # Np × 1
    y_coords = ones(Np, 1)      # Np × 1，全是1

    P = AVariable(0.0, (x_coords, y_coords))  # pressure 
    r = AVariable(0.0, (x_coords, y_coords))  # residual 

    fd_register_unknown!(P)

    bian_in_domain = matrix_setdiff(bianx_quanju_list, boundary_inelem_temp)

    bqjl_in_domain_idx = fast_local_to_global_mapping!(bian_in_domain, bianx_quanju_list) #返回内部边的全局索引

    for k in bqjl_in_domain_idx  #需要Np-Np_boundary个方程组
        #边元-->连接单元-->连接单元布局边远顺序，映射全局
        i = bianx_quanju_list[k,1]; j = bianx_quanju_list[k,2]

        n1_elem = bianx_in_elems_mat[i, j]; Lcmv1 = local_con_matrix_vector[n1_elem]
        m1_local_idx_list = fast_local_to_global_mapping!([i j], Lcmv1) #这个地方应该是Lcm1中每一个
        m1_local_idx = m1_local_idx_list[1]

        n2_elem = bianx_in_elems_mat[j, i]; Lcmv2 = local_con_matrix_vector[n2_elem]
        m2_local_idx_list = fast_local_to_global_mapping!([j i], Lcmv2)
        m2_local_idx = m2_local_idx_list[1]

        C1 = hydraulic_conductance[n1_elem]
        C2 = hydraulic_conductance[n2_elem]

        n1_num = size(Lcmv1,1); n2_num = size(Lcmv2,1)
        Lcmv_to_bql1 = fast_local_to_global_mapping!(Lcmv1, bianx_quanju_list) #表示局部边对应的全局索引
        Lcmv_to_bql2 = fast_local_to_global_mapping!(Lcmv2, bianx_quanju_list)

        Q_eq0 = C1[m1_local_idx, 1] * P[Lcmv_to_bql1[1],1]
        for s = 2:n1_num
            Q_eq0 = Q_eq0 + C1[m1_local_idx, s] * P[Lcmv_to_bql1[s],1]
        end

        Q_eq1 = C2[m2_local_idx, 1] * P[Lcmv_to_bql2[1],1]
        for s = 2:n2_num
            Q_eq1 = Q_eq1 + C2[m2_local_idx, s] * P[Lcmv_to_bql2[s],1]
        end

        Q_eq = 0.0 * P[Lcmv_to_bql1[m1_local_idx], 1] - Q_eq1 - Q_eq0
                
        r[k,1] = paraSrc(k,1) + Q_eq
    end

    #
    # B.C.:

    for k in boundary_global_indices
        #接下来返回边界全局编号, 边界条件
        i = bianx_quanju_list[k,1]; j = bianx_quanju_list[k,2]
        if boundary_index_temp[i,j] == 1
            r[k,1] = P[k,1] - paraboundary1(k,1)
        elseif boundary_index_temp[i,j] == 2
            r[k,1] = P[k,1] - paraboundary2(k,1)
        elseif boundary_index_temp[i,j] == 3
            r[k,1] = P[k,1] - paraboundary3(k,1)
        elseif boundary_index_temp[i,j] == 4
            println("boundary4: = ", i,"  ",j)
            r[k,1] = P[k,1] - paraboundary4(k,1)
        else
            println("边界设计有问题!")
        end
    end


    #
    raw_soln = fd_solve(r)
    pres_soln = fd_get_solution(P, raw_soln) 
    #
    # println("pres_soln = ", pres_soln)
    println("pres_soln_daxiao = ", size(pres_soln,1))
    return (P=P, r=r, pres_soln=pres_soln)

end


# nx_fin = 420; ny_fin = 420
# res = generate_porous_medium(nx_fin,ny_fin,400, 1.1) #560 560
# qiege = get_qiegedanyuan(res.cellDelaunay, res.xs, res.ys, res.sta_cellType, res.Num_Delaunay, res.circles, res.elems, nx_fin,ny_fin)
# infor = Information_Matrix!(qiege.exp_cellType_fin, qiege.xsC, qiege.ysC, qiege.XsC, qiege.YsC, qiege.circles_temp, nx_fin, ny_fin, qiege.elems_temp, qiege.R, qiege.center1)
# #hydraulic_conductance = get_all_porousMedium_subDomain(infor.XsC, infor.YsC, infor.exp_cellType_fin_temp, infor.local_con_matrix_vector, infor.bianx_in_elems_mat, infor.boundary_index_temp, infor.elem_list)
# # solveing = dsolve_pressure(hydraulic_conductance, infor.boundary_global_indices, infor.boundary_index_temp, infor.boundary_inelem_temp, infor.bianx_in_elems_mat, infor.local_con_matrix_vector, infor.bianx_quanju_list)

#----------------------------------------------Stokes-------------------------------------------
function solve_StokesFlow_new!(cellType, xsC, ysC, iPresBdry, elem_k, demo_subdomain, vx_bainx_vel, vy_bainx_vel)
    #外围是边界
    #Celltype 包含周围一圈的虚拟中点，如对于ux 1, 2...Nx, Nx+1, 共 Nx+1
    # To solve the Stokes flow: 
    # ∇⋅v = 0 in Ω ⊂ Ω0 = [a0,a1]x[b0,b1]  
    # μ∇⋅∇v - ∇p + f = 0 in Ω ⊂ Ω0  
    #   B.C.: p = pb and u⋅τ=0 on Γₚ ⊂ ∂Ω (the inflow/outflow boundary)
    #         u = 0 (i.e. u⋅n=0 and u⋅τ=0) on Γᵥ ⊂ ∂Ω (the no-flow/no-slip boundary)
    #

    paraMu = 1.0 # viscosity μ 
    paraSrcX = 0.0;  paraSrcY = 0.0  # f = [fx, fy] 
    # paraSrcX = (iC,jC) -> 0.0;  paraSrcY = (iC,jC) -> 0.0  # f = [fx, fy] 
    # be sure ∂Ω does NOT overlap with ∂Ω0 
    #
    # (cellType, xsC, ysC) = ex_get_domain_case2_three_solid_circles()
    # (cellType, xsC, ysC) = ex_get_domain_case1_simple_straignt_channel()
    # presBdry = (iC,jC) -> 1.0*(cellType[iC,jC]==iPresBdry) 
    presBdry = zeros(size(cellType))
    for  k = 1:size(iPresBdry,1)
        presBdry = presBdry + iPresBdry[k] * (cellType .== k)
    end
    #分配celltype是根据点逆时针连线右侧来判断, 分界线是两点之间的连线

    # presBdry = (iC,jC) -> 100.0*(cellType[iC,jC]==1) 
    # presBdry = (iC,jC) -> 1.0*(cellType[iC,jC]==3) 
    fig_cellType = Plots.contourf(xsC, ysC, cellType', color=:viridis, plot_title="domain cellType"); display(fig_cellType)
    # return fig_cellType 
    hx = xsC[2] - xsC[1];  hy = ysC[2] - ysC[1]  # assuming uniform mesh 
    xsN = vcat(xsC[1]-(hx/2.0), xsC .+ (hx/2.0)) #网格边 所有x剖分边 Nx + 2条，比celltype 中心节点多1
    ysN = vcat(ysC[1]-(hy/2.0), ysC .+ (hy/2.0)) 
    XsC = [x for x in xsC, y in ysC];  YsC = [y for x in xsC, y in ysC]  # for cell center data 
    XsX = [x for x in xsN, y in ysC];  YsX = [y for x in xsN, y in ysC]  # for x-edge center data 
    XsY = [x for x in xsC, y in ysN];  YsY = [y for x in xsC, y in ysN]  # for y-edge center data 
    XsN = [x for x in xsN, y in ysN];  YsN = [y for x in xsN, y in ysN]  # for nodal data 节点数据 共Nx + 2 * Nx + 2
    #
    p = AVariable(0.0, (XsC,YsC))   # pressure 初值以及变量自由度
    enabled::Array{Bool} = (cellType.==0)

    #定义计算变量域内点
    # @assert all(.!enabled[[1,2,end-1,end],:]) && all(.!enabled[:,[1,2,end-1,end]])  # ∂Ω has distance of 2 from ∂Ω0 
    @assert all(.!enabled[[1,end],:]) && all(.!enabled[:,[1,end]])  # ∂Ω has distance of 1 from ∂Ω0 
    fd_restrict_unknown!(p, enabled)
    #
    velx = AVariable(0.0, (XsX,YsX))  # x-component of velocity 
    isEdgX0 = falses(size(XsX)); isEdgX1 = falses(size(XsX)); isEdgX = falses(size(XsX)) #包含虚拟节点边，而之前泊松问题不包含
    isEdgX0[2:end-1,:] = enabled[2:end,:] .& (.!enabled[1:end-1,:])  # boundary x-edges facing west 现在应该属于中间部分，所以横轴要去掉上下进行赋值
    isEdgX1[2:end-1,:] = enabled[1:end-1,:] .& (.!enabled[2:end,:])  # boundary x-edges facing east   
    isEdgX[2:end-1,:]  = enabled[2:end,:] .& enabled[1:end-1,:]      # interior x-edges 
    enabledEgX = isEdgX .| isEdgX0 .| isEdgX1 #此时u_x，u_y虽然包含自己对应的虚拟节点边，但是对应位置的布尔变量是0. 
    fd_restrict_unknown!(velx, enabledEgX) 
    fig_cellType = Plots.heatmap(isEdgX0');
    #             save_plot_with_dir(fig_cellType, "generate_porous_medium/boundary.png", folder="results")
    #
    vely = AVariable(0.0, (XsY,YsY))  # y-component of velocity 
    isEdgY0 = falses(size(XsY)); isEdgY1 = falses(size(XsY)); isEdgY = falses(size(XsY)) #同理包含相应虚拟节点边的索引
    isEdgY0[:,2:end-1] = enabled[:,2:end] .& (.!enabled[:,1:end-1])  # boundary y-edges facing south #所以要
    isEdgY1[:,2:end-1] = enabled[:,1:end-1] .& (.!enabled[:,2:end])  # boundary y-edges facing north 
    isEdgY[:,2:end-1]  = enabled[:,2:end] .& enabled[:,1:end-1]      # interior y-edges 
    enabledEgY = isEdgY .| isEdgY0 .| isEdgY1
    fd_restrict_unknown!(vely, enabledEgY) 
    #
    fd_register_unknown!([p,velx,vely])
    #

    #注意每一项涉及到边界的都需要处理，刚度矩阵单独处理，组装总刚矩阵，届时最终求解时可以删除边界处理

    # Step 1: set the equation for the conservation law of mass (∇⋅v = 0) 
    dvxdx = (velx[2:end,1:end] - velx[1:end-1,1:end]) / hx 
    dvydy = (vely[1:end,2:end] - vely[1:end,1:end-1]) / hy  
    @assert size(dvxdx)==size(p) && size(dvydy)==size(p)  #因为其实对应的是压力的方程
    eqMass = dvxdx + dvydy 
    # Step 2: set the equation for the conservation law of momentum (μ∇⋅∇v - ∇p + f = 0) 
    # 2a) correct to enforce dvxdx=0 and dvydy=0 on inflow/outflow bdry:  
    for i in 1:length(xsC)
        for j in 1:length(ysC) 
            if !enabled[i,j] #bianjie 虚拟中心的导数均为0, u的x方向，v的y方向, 以及颗粒内部
                dvxdx[i,j] = 0.0*velx[i,j]  # <== DD ASSUMPITON HERE for inflow/outflow bdry cond! # 将x方向速度梯度设为0
                dvydy[i,j] = 0.0*vely[i,j]  # <== DD ASSUMPITON HERE for inflow/outflow bdry cond! # 将y方向速度梯度设为0
                # <-- Do not affect for solid-fluid no-slip boundary condition. 
            end
        end
    end
    #
    # 2b) calculate dvxdy and dvydx with bdry treatment:  
    dvxdy =  (velx[1:end,2:end] - velx[1:end,1:end-1]) / hy
    dvydx =  (vely[2:end,1:end] - vely[1:end-1,1:end]) / hx
    for i in 2:length(xsN)-1  # for interior nodes nly  针对待求节点索引 u对y而言，x保持不变,y的中心索引k下面对应的第k条y轴剖分边, 所有索引均从1开始
        for j in 2:length(ysN)-1  
            if cellType[i-1,j-1]>=0 && cellType[i,j-1]>=0 && (cellType[i-1,j]<0 || cellType[i,j]<0) #下边两个单元中心是流体，上边至少有一个是固体
                dvxdy[i,j-1] = (0.0 - velx[i,j-1]) / (hy/2.0)  #velx[i,j-1] 是流体内部距离固体最近点的速度
            elseif cellType[i-1,j]>=0 && cellType[i,j]>=0 && (cellType[i-1,j-1]<0 || cellType[i,j-1]<0)
                dvxdy[i,j-1] = (velx[i,j] - 0.0) / (hy/2.0) 
            elseif cellType[i-1,j]>0 || cellType[i,j]>0 || cellType[i-1,j-1]>0 || cellType[i,j-1]>0 #任何单元是流入流出边界（>0）
                dvxdy[i,j-1] = 0.0*velx[i,j] 
            end
            if cellType[i-1,j-1]>=0 && cellType[i-1,j]>=0 && (cellType[i-1,j]<0 || cellType[i,j]<0)
                dvydx[i-1,j] = (0.0 - vely[i-1,j]) / (hx/2.0) 
            elseif cellType[i,j-1]>=0 && cellType[i,j]>=0 && (cellType[i-1,j-1]<0 || cellType[i-1,j]<0)
                dvydx[i-1,j] = (vely[i,j] - 0.0) / (hx/2.0) 
            elseif cellType[i,j-1]>0 || cellType[i,j]>0 || cellType[i-1,j-1]>0 || cellType[i-1,j]>0
                dvydx[i-1,j] = 0.0*vely[i,j] 
            end
        end
    end
    # 
    # 2c) calculate dpdx and dpdy with bdry treatment (need for inflow/outflow bdry only):  
    
    dpdx = (p[2:end,1:end] - p[1:end-1,1:end]) / hx 
    for i in 2:length(xsN)-1 
        for j in 1:length(ysC) 
            if isEdgX0[i,j]
                @assert !isEdgX1[i,j] 
                # println("pressure = ", presBdry(i-1,j))
                # println("isEdgX0[i,j]: (i,j) = ", (i,j), " cellType[i,j] = ", cellType[i,j], " cellType[i-1,j] = ", cellType[i-1,j])  # dbg
                dpdx[i-1,j] = (p[i,j] - presBdry[i-1,j]) / (hx/2.0) #都是向前, 压力求导在剖分边界处
            elseif isEdgX1[i,j] 
                @assert !isEdgX0[i,j]
                # println("isEdgX1[i,j]: (i,j) = ", (i,j), " cellType[i,j] = ", cellType[i,j], " cellType[i-1,j] = ", cellType[i-1,j])  # dbg
                # println(" presBdry(i,j) = ", presBdry(i,j)) # dbg 
                dpdx[i-1,j] = (presBdry[i,j] - p[i-1,j]) / (hx/2.0)
            end 
        end
    end
    dpdy = (p[1:end,2:end] - p[1:end,1:end-1]) / hy 
    for i in 1:length(xsC) 
        for j in 2:length(ysN)-1 
            if isEdgY0[i,j]
                @assert !isEdgY1[i,j] 
                dpdy[i,j-1] = (p[i,j] - presBdry[i,j-1]) / (hy/2.0)
            elseif isEdgY1[i,j] 
                @assert !isEdgY0[i,j]
                dpdy[i,j-1] = (presBdry[i,j] - p[i,j-1]) / (hy/2.0)
            end 
        end
    end
    #
    # 2d) calculate eqMomtX and eqMomtY with bdry treatment 
    dvxdxx = (dvxdx[2:end,1:end] - dvxdx[1:end-1,1:end]) / hx  # no need for dvxdxy and dvxdyx 
    dvxdyy = (dvxdy[1:end,2:end] - dvxdy[1:end,1:end-1]) / hy 
    dvydxx = (dvydx[2:end,1:end] - dvydx[1:end-1,1:end]) / hx  
    dvydyy = (dvydy[1:end,2:end] - dvydy[1:end,1:end-1]) / hy 
    eqMomtX = paraMu * (dvxdxx[1:end,2:end-1] + dvxdyy[2:end-1,1:end]) - dpdx[1:end,2:end-1] + paraSrcX  
    eqMomtY = paraMu * (dvydxx[1:end,2:end-1] + dvydyy[2:end-1,1:end]) - dpdy[2:end-1,1:end] + paraSrcY  
    #
    # 2e) correct for the no-slip boundaries: 
    @assert size(isEdgX)==(length(xsN), length(ysC))
    for i in 2:length(xsN)-1 
        for j in 2:length(ysC)-1  
            if isEdgX0[i,j] 
                @assert cellType[i,j]==0 && cellType[i-1,j]!=0
                if cellType[i-1,j]<0
                    eqMomtX[i-1,j-1] = velx[i,j] - 0.0  # velx = 0.0 on no-slip edgeX  都减1表示外围边界 通过直接设置动量方程来强制速度为零
                end
            end 
            if isEdgX1[i,j] 
                @assert cellType[i-1,j]==0 && cellType[i,j]!=0
                if cellType[i,j]<0
                    eqMomtX[i-1,j-1] = velx[i,j] - 0.0  # velx = 0.0 on no-slip edgeX  
                end
            end 
        end
    end
    @assert size(isEdgY)==(length(xsC), length(ysN)) #颗粒无滑移边界条件
    for i in 2:length(xsC)-1  
        for j in 2:length(ysN)-1 
            if isEdgY0[i,j] 
                @assert cellType[i,j]==0 && cellType[i,j-1]!=0
                if cellType[i,j-1]<0
                    eqMomtY[i-1,j-1] = vely[i,j] - 0.0  # vely = 0.0 on no-slip edgeY  
                end
            end 
            if isEdgY1[i,j] 
                @assert cellType[i,j-1]==0 && cellType[i,j]!=0
                if cellType[i,j]<0
                    eqMomtY[i-1,j-1] = vely[i,j] - 0.0  # vely = 0.0 on no-slip edgeY  b = 0 - bianjiezhi, A = -1
                end
            end             
        end
    end
    # 
    # println("size(enabled) = ", size(enabled)) # dbg 
    # println("size(enabledEgX) = ", size(enabledEgX)) # dbg 
    # println("size(enabledEgY) = ", size(enabledEgY)) # dbg 
    fd_restrict_residual!(eqMass, enabled) 
    fd_restrict_residual!(eqMomtX, enabledEgX[2:end-1,2:end-1])  #为什么这样，是因为离散区域包含物理计算区域，删除这部分不影响刚度矩阵
    fd_restrict_residual!(eqMomtY, enabledEgY[2:end-1,2:end-1]) 
    #
    raw_soln = fd_solve([eqMass, eqMomtX, eqMomtY])
    p_soln = fd_get_solution(p, raw_soln) 
    vx_soln = fd_get_solution(velx, raw_soln); vy_soln = fd_get_solution(vely, raw_soln);  
    println("size = ", size(p_soln), "    ", size(vx_soln), "    ", size(vy_soln)) 
    #
    # calculate total flow rate in each of the inlets/outlets, 因为现在是Stokes 方程而不是Darcy, 所以考虑u, v即可
    numInOutLets = maximum(cellType) #返回流入流出max个类型编号，创建max x 1的数组用于储存, 流入流出是根据设计边界来的而不是根据左右猜测
    flowOutRates = zeros(numInOutLets)

    vx_boundary_local_bool = falses(size(XsX)); vy_boundary_local_bool = falses(size(XsY))
    iMin0 = demo_subdomain[1,1]; jMin0 = demo_subdomain[1,2]
    for i in 2:length(xsN)-1 
        for j in 2:length(ysC)-1  
            if isEdgX0[i,j] 
                if cellType[i-1,j]>0
                    vx_bainx_vel[iMin0+i-1, jMin0+j-1] = vx_soln[i,j]
                    flowOutRates[cellType[i-1,j]] += -vx_soln[i,j]*hy #流出流量 = 速度 × 横截面积 西向边界的正向速度是向右，但流出是向左，所以取负值
                end #流入应该等于左侧单元的流出，所以往左平移1格
            end 
            if isEdgX1[i,j] 
                if cellType[i,j]>0
                    vx_bainx_vel[iMin0+i-1, jMin0+j-1] = vx_soln[i,j]
                    flowOutRates[cellType[i,j]] += vx_soln[i,j]*hy 
                end
            end 
        end
    end
    for i in 2:length(xsC)-1  
        for j in 2:length(ysN)-1 
            if isEdgY0[i,j] 
                if cellType[i,j-1]>0
                    vy_bainx_vel[iMin0+i-1, jMin0+j-1] = vy_soln[i,j]
                    flowOutRates[cellType[i,j-1]] += -vy_soln[i,j]*hx 
                end
            end 
            if isEdgY1[i,j] 
                if cellType[i,j]>0
                    vy_bainx_vel[iMin0+i-1, jMin0+j-1] = vy_soln[i,j]
                    flowOutRates[cellType[i,j]] += vy_soln[i,j]*hx 
                end
            end             
        end
    end 
    # 
    # plot pressure: 
    min_val, max_val = extrema(p_soln)
    zero_region_levels = range(-1e-4 * max_val, 1.e-4 * max_val, length=5)  # 0附近150个级别
    positive_levels = range(1.e-4* max_val, max_val, length=30)             # 正区域75个级别  
    negative_levels = range(-1 * max_val, -1.e-4*max_val, length=5)            # 负区域75个级别
    custom_levels = vcat(negative_levels, zero_region_levels[2:end-1], positive_levels)
    fig_pres = Plots.contourf(xsC, ysC, p_soln', color=:viridis, plot_title="pressure solution", 
                         levels=custom_levels,  # 使用非均匀级别
                         clims=extrema(p_soln),
                         linewidth=0.8,
                         colorbar=true,
                         colorbar_title="Pressure",
                         aspect_ratio=:equal)
    display(fig_pres)

    # fig_pres = Plots.contourf(xsC, ysC, p_soln', color=:viridis, plot_title="pressure solution",levels=20,
    #                      clims=extrema(p_soln),
    #                      linewidth=0.8,
    #                      colorbar=true,
    #                      colorbar_title="Pressure",
    #                      aspect_ratio=:equal)
    # display(fig_pres)
    save_plot_with_dir(fig_pres, "generate_porous_medium/pre_num_0001/fig_pre_$elem_k.png", folder="results")
    # plot velocity: 
    vx = 0.5*(vx_soln[2:end-1,1:end-1] + vx_soln[2:end-1,2:end]) #网格节点处
    vy = 0.5*(vy_soln[1:end-1,2:end-1] + vy_soln[2:end,2:end-1]) 

    vx_C = 0.5*(vx_soln[1:end-1,1:end] + vx_soln[2:end,1:end]) #网格节点处
    vy_C = 0.5*(vy_soln[1:end,1:end-1] + vy_soln[1:end,2:end]) 

    XsNd = XsN[2:end-1,2:end-1]; YsNd = YsN[2:end-1,2:end-1]
    @assert size(vx)==size(vy) && size(vx)==size(XsNd) && size(vy)==size(YsNd)  
    
   # 调节稀疏性：步长越大越稀疏
    stride = 3  # 尝试 3-8 之间的值
    # 创建降采样索引
    i_idx = 1:stride:size(XsNd, 1)
    j_idx = 1:stride:size(XsNd, 2)
    # 应用降采样
    XsNd_sub = XsNd[i_idx, j_idx]
    YsNd_sub = YsNd[i_idx, j_idx]
    vx_sub = vx[i_idx, j_idx]
    vy_sub = vy[i_idx, j_idx]

    magnitudes = sqrt.(vx_sub.^2 + vy_sub.^2) 
    # relativeSizes = magnitudes./maximum(magnitudes) 
    maxArrowLength = 0.2*0.2   # domain [0,1] Umax = 10 then 10 should be ——> 0.2 
    #Only need to define the maxArrowLength to control the entire plot
    scale = maximum(magnitudes)/maxArrowLength
    fig_vel = Plots.quiver(XsNd_sub, YsNd_sub, quiver=(vx_sub/scale, vy_sub/scale), color=:black, plot_title="velocity solution") 
    display(fig_vel) 
    save_plot_with_dir(fig_vel, "generate_porous_medium/vel_num_0001/fig_vel_$elem_k.png", folder="results")
    println("outinflow = ", flowOutRates)

    open("D:/Juali_code/Stokesflow/ourtest/ceshi30/0001/inoutput_0001.txt", "a") do file
        println(file, "elem_k = $elem_k")
        println(file, join(flowOutRates, "\t"))
        println(file)  # 最后换行
        println(file)
    end

    # dvxdyy=dvxdyy, dvxdy=dvxdy, dpdx=dpdx,  # tmp return 
    return (vx_C = vx_C, vy_C = vy_C, flowOutRates=flowOutRates, raw_soln=raw_soln, cellType=cellType, enabled=enabled, enabledEgX=enabledEgX, enabledEgY=enabledEgY, 
    p=p, velx=velx, vely=vely, eqMass=eqMass, eqMomtX=eqMomtX, eqMomtY=eqMomtY, p_soln=p_soln, vx_soln=vx_soln, vy_soln=vy_soln, 
    fig_pres=fig_pres, fig_vel=fig_vel, XsX = XsX, XsY = XsY, XsN = XsN, YsN = YsN, vx_boundary_local_bool = vx_boundary_local_bool, 
    vy_boundary_local_bool = vy_boundary_local_bool, vx_bainx_vel=vx_bainx_vel, vy_bainx_vel=vy_bainx_vel)      
end

function back_orign_info(XsC, YsC, Pres, elem_list, hydraulic_conductance, boundary_index_temp, bianx_in_elems_mat, exp_cellType_fin_temp, local_con_matrix_vector, boundary_i_to_j, bianx_quanju_list)

    xsC = XsC[:,1]; ysC = YsC[1,:]
    hx = xsC[2] - xsC[1];  hy = ysC[2] - ysC[1]  # assuming uniform mesh 
    xsN = vcat(xsC[1]-(hx/2.0), xsC .+ (hx/2.0)) #网格边 所有x剖分边 Nx + 2条，比celltype 中心节点多1
    ysN = vcat(ysC[1]-(hy/2.0), ysC .+ (hy/2.0))  
    XsX = [x for x in xsN, y in ysC];  YsX = [y for x in xsN, y in ysC]  # for x-edge center data 
    XsY = [x for x in xsC, y in ysN];  YsY = [y for x in xsC, y in ysN]  # for y-edge center data
    XsN = [x for x in xsN, y in ysN];  YsN = [y for x in xsN, y in ysN]  # for nodal data 节点数据 共Nx + 2 * Nx + 2

    println("XsX = ", size(XsX), "   ", size(XsY))
    open("D:/Juali_code/Stokesflow/ourtest/ceshi30/0001/inoutput_0001.txt", "w") do file
        print(file, "所属单元:            ")
        print(file, "单元流量:            ")
        println(file)
    end

    Np = size(bianx_quanju_list,1); N = size(local_con_matrix_vector,1)
    Pre_mat = zeros(Np, Np)
    Pre_vec_local = Vector{Vector{Float64}}(undef, N)
    pre_value = zeros(size(exp_cellType_fin_temp))

    vx_bainx_vel1 = zeros(size(XsX))
    vy_bainx_vel1 = zeros(size(XsY))

    vx_valueC = zeros(size(XsC))
    vy_valueC = zeros(size(XsC))

    for i = 1:N
        Pre_vec_local[i] = [0,0,0,0]  # 或其他初始化方式
    end

    for k = 1:Np #k表示第k边远, 第k边元的值矩阵
        i = bianx_quanju_list[k,1]; j = bianx_quanju_list[k,2] #i，j表示边点i-->j
        Pre_mat[i, j] = Pres[k]
    end
    Pre_mat = Pre_mat + Pre_mat'

    open("D:/Juali_code/Stokesflow/ourtest/ceshi30/0001/elem_pre_1000.txt", "w") do file
        print(file, "单元:")
        print(file, "单元边压力 P_{i,j}:")
        println(file)
    end

    for k in elem_list #k表示第k个单元压力索引
        temp1 = local_con_matrix_vector[k]
        temp = Vector{Float64}()  # 创建浮点数空数组

        for m = 1:size(temp1, 1)
            value = Pre_mat[temp1[m, 1], temp1[m, 2]]
            push!(temp, value)  # 使用 push! 添加
        end
        Pre_vec_local[k] = temp

        open("D:/Juali_code/Stokesflow/ourtest/ceshi30/0001/elem_pre_1000.txt", "a") do file
            println(file, "单元-$k")
            println(file, join(Pre_vec_local[k], "\t"))
            println(file)
        end

        sbRes = get_porousMedium_subDomain(XsC, YsC, exp_cellType_fin_temp, local_con_matrix_vector, bianx_in_elems_mat, boundary_index_temp, k)
        # iPresBdry = Pre_vec_local[bianx_in_elems_mat[31,21]]
        solve_num_value = solve_StokesFlow_new!(sbRes.locCellType, sbRes.sbXsC[:,1], sbRes.sbYsC[1,:], temp, k, sbRes.demo_subdomain, vx_bainx_vel1, vy_bainx_vel1)

        vx_bainx_vel1 = solve_num_value.vx_bainx_vel
        vy_bainx_vel1 = solve_num_value.vy_bainx_vel
        psub_soln = solve_num_value.p_soln
        uxC_soln = solve_num_value.vx_C
        vyC_soln = solve_num_value.vy_C

        #将局部数据返回到原始数据上,速度的也要----------------------------------------shuzhi-------------------------------
        println("size1 = ", size(psub_soln), "    ", size(sbRes.demo_subdomain))
        pre_value[exp_cellType_fin_temp .== k] = psub_soln[sbRes.locCellType .== 0]
        vx_valueC[exp_cellType_fin_temp .== k] = uxC_soln[sbRes.locCellType .== 0]
        vy_valueC[exp_cellType_fin_temp .== k] = vyC_soln[sbRes.locCellType .== 0]


    end
    println("size2 = ", size(XsC), "   ", size(pre_value))


    println("size2 = ", size(XsC), "   ", size(vx_valueC))

    println("size3 = ", size(vx_bainx_vel1), "    ", size(vy_bainx_vel1))
    # 创建非均匀级别，在0附近加密，可视化
    min_val, max_val = extrema(pre_value)

    # 在0附近创建密集级别，其他地方稀疏
    zero_region_levels = range(-1e-4 * max_val, 1.e-3 * max_val, length=30)  # 0附近150个级别
    positive_levels = range(1.e-3* max_val, max_val, length=60)             # 正区域75个级别  
    negative_levels = range(min_val, -0.1, length=60)            # 负区域75个级别

    # 合并所有级别（去掉重复的0点）
    custom_levels = vcat(negative_levels, zero_region_levels[2:end-1], positive_levels)

    fig_presfin = Plots.contourf(XsC[:,1], YsC[1,:], pre_value', 
                            color=:viridis,
                            plot_title="pressure solution", 
                            levels=custom_levels,  # 使用非均匀级别
                            clims=extrema(pre_value),
                            linewidth=0.8,
                            colorbar=true,
                            colorbar_title="Pressure",
                            aspect_ratio=:equal)

    # fig_presfin = Plots.contourf(XsC[:,1], YsC[1,:], pre_value', color=:viridis, plot_title="pressure solution", levels=300,
    #                      clims=extrema(pre_value),
    #                      linewidth=0.8,
    #                      colorbar=true,
    #                      colorbar_title="Pressure",
    #                      aspect_ratio=:equal) 
    display(fig_presfin)
    save_plot_with_dir(fig_presfin, "generate_porous_medium/pressure_final0000.png", folder="results")


    vx = 0.5*(vx_bainx_vel1[2:end-1,1:end-1] + vx_bainx_vel1[2:end-1,2:end]) #网格节点处
    vy = 0.5*(vy_bainx_vel1[1:end-1,2:end-1] + vy_bainx_vel1[2:end,2:end-1]) 
#     XsNd = XsN[2:end-1,2:end-1]; YsNd = YsN[2:end-1,2:end-1]
#     @assert size(vx)==size(vy) && size(vx)==size(XsNd) && size(vy)==size(YsNd)  
    
   # 调节稀疏性：步长越大越稀疏
    stride = 10  # 尝试 3-8 之间的值
    # 创建降采样索引
    i_idx = 1:stride:size(XsC, 1)
    j_idx = 1:stride:size(XsC, 2)
    # 应用降采样
    XsNd_sub = XsC[i_idx, j_idx]
    YsNd_sub = YsC[i_idx, j_idx]
    vx_valueC_sub = vx_valueC[i_idx, j_idx]
    vy_valueC_sub = vy_valueC[i_idx, j_idx]
    
    magnitudes = sqrt.(vx_valueC_sub.^2 + vy_valueC_sub.^2) 
    # XsNd = XsN[2:end-1,2:end-1]; YsNd = YsN[2:end-1,2:end-1]
    # relativeSizes = magnitudes./maximum(magnitudes) 
    maxArrowLength = 0.2*0.2   # domain [0,1] Umax = 10 then 10 should be ——> 0.2 
    #Only need to define the maxArrowLength to control the entire plot
    scale = maximum(magnitudes)/maxArrowLength
    scale_factor = 1
    fig_vel = Plots.quiver(XsNd_sub, YsNd_sub, quiver=(vx_valueC_sub/(scale*scale_factor), vy_valueC_sub/(scale*scale_factor)), color=:blue,
                      linewidth=0.3, #增加线宽让箭头更明显
                      arrow=:line,
                      arrowsize=0.05, #控制箭头大小
                      arrowheadlength=0.06,
                      arrowheadwidth=0.05,
                    #   alpha=0.8,                    # 透明度
                      plot_title="Velocity Vectors",
                      framestyle=:box,
                      background_color=:white,  # 设置背景为纯白
                      xlims=(minimum(XsNd_sub), maximum(XsNd_sub)),
                      ylims=(minimum(YsNd_sub), maximum(YsNd_sub)),
                      aspect_ratio=:equal,
                      size=(800, 600))

    display(fig_vel)

    save_plot_with_dir(fig_vel, "generate_porous_medium/vel_num_0001/fig_vel_all.png", folder="results")
    # sbRes = get_porousMedium_subDomain(XsC, YsC, exp_cellType_fin_temp, local_con_matrix_vector, bianx_in_elems_mat, boundary_index_temp, bianx_in_elems_mat[31,21])
    # iPresBdry = Pre_vec_local[bianx_in_elems_mat[31,21]]
    # println("value: = ", iPresBdry)
    # println("u = ", bianx_in_elems_mat[24,39], "   ", local_con_matrix_vector[bianx_in_elems_mat[24,39]])
    # solve_StokesFlow_new(sbRes.locCellType, sbRes.sbXsC[:,1], sbRes.sbYsC[1,:], iPresBdry)
    
    
    return Pre_vec_local
end

nx_fin = 420; ny_fin = 420
res = generate_porous_medium(nx_fin,ny_fin,400, 1.1) #560 560
qiege = get_qiegedanyuan(res.cellDelaunay, res.xs, res.ys, res.sta_cellType, res.Num_Delaunay, res.circles, res.elems, nx_fin,ny_fin)
infor = Information_Matrix!(qiege.exp_cellType_fin, qiege.xsC, qiege.ysC, qiege.XsC, qiege.YsC, qiege.circles_temp, nx_fin, ny_fin, qiege.elems_temp, qiege.R, qiege.center1)
hydraulic_conductance = get_all_porousMedium_subDomain(infor.XsC, infor.YsC, infor.exp_cellType_fin_temp, infor.local_con_matrix_vector, infor.bianx_in_elems_mat, infor.boundary_index_temp, infor.elem_list)
solveing = dsolve_pressure(hydraulic_conductance, infor.boundary_global_indices, infor.boundary_index_temp, infor.boundary_inelem_temp, infor.bianx_in_elems_mat, infor.local_con_matrix_vector, infor.bianx_quanju_list)
back_vel = back_orign_info(infor.XsC, infor.YsC, solveing.pres_soln, infor.elem_list, hydraulic_conductance, infor.boundary_index_temp, infor.bianx_in_elems_mat, infor.exp_cellType_fin_temp, infor.local_con_matrix_vector, infor.boundary_inelem_temp, infor.bianx_quanju_list)