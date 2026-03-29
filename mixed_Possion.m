addpath(genpath('MAC_fun'));

try
    out = ex_PoissonEqu_implement5();
catch ME
    fprintf('❌ 计算失败！错误信息:\n');
    fprintf('   错误位置: %s (第 %d 行)\n', ME.stack(1).name, ME.stack(1).line);
    fprintf('   错误信息: %s\n', ME.message);
end
%-------------------------------------------
rmpath(genpath('MAC_fun'))
%------------------------------------

function out = ex_PoissonEqu_implement5()
% Solve Poisson in irregular domain Omega via mixed form:
%   div(v) = f,  v = -grad(p)
% Unknowns: p (cell-centered), velx (x-edges), vely (y-edges)
% Domain: circle minus square hole, fully inside [0,1]^2

    % ---- BC / source (vectorized) ----
    presBdry = @(x,y) 10.0 + 0.*x + 0.*y;
    paraSrc  = @(x,y) 1000.0 + 0.*x + 0.*y;

    % ---- grid ----
    nx = 10; ny = 10;
    xs = linspace(0,1,nx+1);
    ys = linspace(0,1,ny+1);

    xsC = 0.5*(xs(1:end-1) + xs(2:end));   % cell centers, length nx
    ysC = 0.5*(ys(1:end-1) + ys(2:end));   % length ny
    [Xs, Ys] = ndgrid(xsC, ysC);           % size nx-by-ny

    % ---- inDomain mask on cell centers ----
    enabled = ((Xs-0.5).^2 + (Ys-0.5).^2) < (0.45^2) ...
              & ~(abs(Xs-0.5) < 0.2 & abs(Ys-0.5) < 0.2);

    % ---- unknown: pressure p on cell centers ----
    p = AVariable_from_function(@(X,Y) 0.*X + 0.*Y, {Xs, Ys});
    fd_restrict_unknown_inplace(p, enabled);

    % ---- unknown: velx on x-edges between cell centers (i=1..nx-1, j=1..ny) ----
    % grid for velx uses Xs(2:end,:) / Ys(2:end,:) as in Julia
    Xx = Xs(2:end, :); %选择内部，这样可以和isEdgX0呼应，数量与未知一样，并没有边界条件,其实是边上的值
    Yx = Ys(2:end, :);
    velx = AVariable_from_function(@(X,Y) 0.*X + 0.*Y, {Xx, Yx});  % size (nx-1)-by-ny

    isEdgX0 = enabled(2:end, :)   & ~enabled(1:end-1, :);  % boundary edges facing west
    isEdgX1 = enabled(1:end-1, :) & ~enabled(2:end, :);    % boundary edges facing east
    isEdgX  = enabled(2:end, :)   &  enabled(1:end-1, :);  % interior edges
    maskX   = isEdgX | isEdgX0 | isEdgX1;
    fd_restrict_unknown_inplace(velx, maskX);

    % ---- unknown: vely on y-edges between cell centers (i=1..nx, j=1..ny-1) ----
    Xy = Xs(:, 2:end);
    Yy = Ys(:, 2:end);
    vely = AVariable_from_function(@(X,Y) 0.*X + 0.*Y, {Xy, Yy});  % size nx-by-(ny-1)

    isEdgY0 = enabled(:, 2:end)   & ~enabled(:, 1:end-1);  % boundary edges facing south 南
    isEdgY1 = enabled(:, 1:end-1) & ~enabled(:, 2:end);    % boundary edges facing north 北
    isEdgY  = enabled(:, 2:end)   &  enabled(:, 1:end-1);  % interior edges
    maskY   = isEdgY | isEdgY0 | isEdgY1;
    fd_restrict_unknown_inplace(vely, maskY);

    % ---- register unknowns into one global DOF vector ----
    fd_register_unknown_multi_inplace([p, velx, vely]);

    % ---- equations ----
    hx = 1.0/nx;
    hy = 1.0/ny;

    %---------------------------- Step 1-----------------------------
    % Darcy law on x-edges (default interior form)
    eqDarcyX = velx + (p(2:nx, :) - p(1:nx-1, :)) / hx;

    % Darcy law on y-edges
    eqDarcyY = vely + (p(:, 2:ny) - p(:, 1:ny-1)) / hy;

    %---------------------------- Step 2-----------------------------
    % ---- correct Darcy equations on internal boundary edges (Dirichlet p = pb) ----
    xsEdg = 0.5*(xsC(1:end-1) + xsC(2:end));  % length nx-1
    for i = 1:(numel(xsC)-1)
        for j = 1:numel(ysC)
            if isEdgX0(i,j)
                eqDarcyX(i,j) = velx(i,j) + (p(i+1,j) - presBdry(xsEdg(i), ysC(j))) / (hx/2.0);
            elseif isEdgX1(i,j)
                eqDarcyX(i,j) = velx(i,j) + (presBdry(xsEdg(i), ysC(j)) - p(i,j)) / (hx/2.0);
            end
        end
    end

    ysEdg = 0.5*(ysC(1:end-1) + ysC(2:end));  % length ny-1
    for i = 1:numel(xsC)
        for j = 1:(numel(ysC)-1)
            if isEdgY0(i,j)
                eqDarcyY(i,j) = vely(i,j) + (p(i,j+1) - presBdry(xsC(i), ysEdg(j))) / (hy/2.0);
            elseif isEdgY1(i,j)
                eqDarcyY(i,j) = vely(i,j) + (presBdry(xsC(i), ysEdg(j)) - p(i,j)) / (hy/2.0);
            end
        end
    end

    %---------- step 4: set the equation for the conservation law (for all enabled cells)
    % Conservation law on enabled interior cells
    dvxdx = (velx(2:nx-1,   2:ny-1) - velx(1:nx-2, 2:ny-1)) / hx;   % (nx-2)x(ny-2)
    dvydy = (vely(2:nx-1,   2:ny-1) - vely(2:nx-1, 1:ny-2)) / hy;   % (nx-2)x(ny-2)
    src   = paraSrc(Xs(2:end-1,2:end-1), Ys(2:end-1,2:end-1));
    eqConsv = src - dvxdx - dvydy;

    % ---- outer boundary of Omega0 must be outside physical boundary ----
    assert(all(all(~enabled([1 end],:))) && all(all(~enabled(:,[1 end]))), ...
        "Require: physical boundary does NOT overlap the box boundary.");

    % ---- restrict residual rows to active equation locations ----
    fd_restrict_residual_inplace(eqDarcyX, maskX);
    fd_restrict_residual_inplace(eqDarcyY, maskY);
    fd_restrict_residual_inplace(eqConsv, enabled(2:end-1,2:end-1));

    % ---- solve global linear system ----
    raw_soln  = fd_solve([eqConsv, eqDarcyX, eqDarcyY]);
    pres_soln = fd_get_solution(p, raw_soln);

    % ---- plot pressure ----
    fig_pres = figure('Name','pressure solution');
    contourf(xsC, ysC, pres_soln.'); axis equal tight; colorbar;
    title('pressure solution');

    % ---- plot velocity (mapped to nodes by averaging) ----
    vx = fd_get_solution(velx, raw_soln);
    vx = 0.5*(vx(:,1:end-1) + vx(:,2:end));        % -> (nx-1)x(ny-1)

    vy = fd_get_solution(vely, raw_soln);
    vy = 0.5*(vy(1:end-1,:) + vy(2:end,:));        % -> (nx-1)x(ny-1)

    [XsNd, YsNd] = ndgrid(xs(2:end-1), ys(2:end-1)); % (nx-1)x(ny-1)
    assert(isequal(size(vx), size(vy)) && isequal(size(vx), size(XsNd)), "velocity size mismatch.");

    magnitudes = sqrt(vx.^2 + vy.^2);
    maxArrowLength = 0.2*0.2;
    mmax = max(magnitudes(:));
    if mmax == 0
        scale = 1.0;
    else
        scale = mmax / maxArrowLength;
    end

    fig_vel = figure('Name','velocity solution');
    quiver(XsNd, YsNd, vx/scale, vy/scale, 0, 'k'); axis equal tight;
    title('velocity solution');

    % ---- output struct (like Julia NamedTuple) ----
    out = struct();
    out.p = p;
    out.velx = velx;
    out.vely = vely;
    out.pres_soln = pres_soln;
    out.fig_pres = fig_pres;
    out.fig_vel = fig_vel;
end
