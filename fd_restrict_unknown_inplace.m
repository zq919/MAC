function unknown = fd_restrict_unknown_inplace(unknown, enabled_in)
% 把未知量从全量 DOF 收缩到“活跃 DOF 子空间”
%FD_RESTRICT_UNKNOWN_INPLACE Restrict unknown.jac columns by an enabled mask.
% Julia equivalent:
%   unknown.jac = unknown.jac[:, enabled[:]]
%
% Inputs:
%   unknown    : AVariable (object with fields/properties mdata, jac)
%   enabled_in : logical array (same number of elements as unknown.mdata)
%
% Notes:
%   - Requires unknown.jac to be identity (or equal to speye(numel(mdata))).
%   - Julia length(...) corresponds to MATLAB numel(...).

    % ---- check enabled mask ----
    enabled = logical(enabled_in);
    enabled_vec = enabled(:);

    dof = numel(unknown.mdata);

    % ---- assertions (match Julia) ----
    % 1) jac must be identity: nnz( jac != I ) == 0
    I = speye(dof);
    assert(nnz(unknown.jac - I) == 0, ...
        "Assertion failed: unknown.jac must be an identity matrix.");

    % 2) size checks
    assert(size(unknown.jac,1) == dof, ...
        "Assertion failed: size(unknown.jac,1) must equal numel(unknown.mdata).");
    assert(size(unknown.jac,2) == numel(enabled_vec), ...
        "Assertion failed: size(unknown.jac,2) must equal numel(enabled).");

    % ---- restrict columns (keep only enabled DOFs) ----
    unknown.jac = unknown.jac(:, enabled_vec);

end
