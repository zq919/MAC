function residual = fd_restrict_residual_inplace(residual, enabled)
%FD_RESTRICT_RESIDUAL_INPLACE Restrict residual by enabled mask.
% Julia equivalent:
%   residual.mdata = residual.mdata[enabled[:]]
%   residual.jac   = residual.jac[enabled[:], :]
%
% Note:
%   residual.mdata will become 1D (column vector) after restriction.

    enabled_vec = logical(enabled(:));   % enabled[:] in Julia

    % Julia length(residual.mdata) -> MATLAB numel(residual.mdata)
    n = numel(residual.mdata);

    % assertions
    assert(size(residual.jac, 1) == n, ...
        "Assertion failed: size(residual.jac,1) must equal numel(residual.mdata).");
    assert(numel(enabled_vec) == n, ...
        "Assertion failed: numel(enabled) must equal numel(residual.mdata).");

    % restrict: mdata becomes 1D
    residual.mdata = residual.mdata(enabled_vec);
    residual.mdata = residual.mdata(:);          % ensure column vector (optional but recommended)

    % restrict jac rows
    residual.jac = residual.jac(enabled_vec, :);
end