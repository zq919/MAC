function raw_soln = fd_solve(residual)
%FD_SOLVE Solve global linear system from multiple residual AVariables.
%
% Accepts:
%   residual = r              (single AVariable)
%   residual = [r1 r2 r3]     (AVariable object array)
%   residual = [r1; r2; r3]   (AVariable object array)
%   residual = {...}          (cell, optional compatibility)
%
% Solves:
%   (vcat r.jac) * x = vcat(-r.mdata(:))

    flat_residuals = flatten_AVariables(residual);   % -> 1D cell {AVariable,...}
    nR = numel(flat_residuals);
    assert(nR >= 1, "Need at least one residual.");

    matSs = cell(1, nR);
    rhss  = cell(1, nR);

    for k = 1:nR
        r = flat_residuals{k};
        assert(isa(r,'AVariable'), "All residual entries must be AVariable.");

        matSs{k} = r.jac;         % Jacobian block
        rhss{k}  = -r.mdata(:);   % RHS block
    end

    matS = vertcat(matSs{:});
    rhs  = vertcat(rhss{:});

    assert(size(matS,1) == numel(rhs), "Row mismatch: size(matS,1) must equal length(rhs).");
    assert(size(matS,2) == numel(rhs), "Square system expected: size(matS,2) must equal length(rhs).");

    raw_soln = matS \ rhs;
end