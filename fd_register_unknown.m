function fd_register_unknown(unknown, verbose)
% Register a single unknown (AVariable-like).
% Julia:
%   @assert size(unknown.jac,1)==length(unknown.mdata)
%   println("To register 1 scalar unknown without any tuned parameters.")

    if nargin < 2
        verbose = 1;
    end

    % 兼容 AVariable 是 handle 类 或 struct
    if isobject(unknown)
        assert(isprop(unknown,'mdata') && isprop(unknown,'jac'), ...
            "unknown must have properties mdata and jac.");
        mdata = unknown.mdata;
        jac   = unknown.jac;
    elseif isstruct(unknown)
        assert(isfield(unknown,'mdata') && isfield(unknown,'jac'), ...
            "unknown must have fields mdata and jac.");
        mdata = unknown.mdata;
        jac   = unknown.jac;
    else
        error("unknown must be an AVariable object or a struct with fields mdata/jac.");
    end

    % Julia 的 length(array) -> MATLAB 用 numel(array)
    assert(size(jac,1) == numel(mdata), ...
        "Assertion failed: size(unknown.jac,1) must equal numel(unknown.mdata).");

    if verbose >= 1
        fprintf("To register 1 scalar unknown without any tuned parameters.\n");
    end
end