function fd_register_unknown_multi_inplace(unknown, verbose)
%FD_REGISTER_UNKNOWN_MULTI_INPLACE Register multiple unknowns by expanding each
%local jacobian into a global-column jacobian (block placement).
%
% Julia equivalent:
%   fd_register_unknown!(unknown::Union{Vector,Matrix,Tuple})
%
% Input:
%   unknown : cell container / AVariable / AVariable array
%   verbose : optional, default 1
%
% Requirement:
%   aux_flatten(unknown) -> 1D cell array {AVariable,...}

    if nargin < 2
        verbose = 1;
    end

    flat_unknowns = aux_flatten(unknown);     % -> cell {AVariable,...}
    assert(numel(flat_unknowns) >= 1, "Need at least one unknown.");

    nU = numel(flat_unknowns);

    dofs = zeros(1, nU);   % number of dofs (columns of jac)
    doqs = zeros(1, nU);   % number of quantities (rows of jac)

    for iU = 1:nU
        u = flat_unknowns{iU};
        assert(isa(u,'AVariable'), "All flattened entries must be AVariable.");
        doqs(iU) = size(u.jac, 1);
        dofs(iU) = size(u.jac, 2);
    end

    N = sum(dofs);

    for iU = 1:nU
        u = flat_unknowns{iU};
        jac_local = u.jac;

        % new global jac
        u.jac = sparse(doqs(iU), N);

        i_beg = sum(dofs(1:iU-1)) + 1;
        i_end = sum(dofs(1:iU));

        u.jac(:, i_beg:i_end) = jac_local;
    end

    if verbose >= 1
        if nU == 1
            fprintf("To register %d scalar unknown without any tuned parameters.\n", nU);
        else
            % mimic "(length(unknown) physical unknowns)"
            if iscell(unknown)
                nPhys = numel(unknown);
            elseif isa(unknown,'AVariable')
                nPhys = 1;
            else
                nPhys = numel(unknown);
            end
            fprintf("To register %d scalar unknowns (%d physical unknowns) without any tuned parameters.\n", ...
                nU, nPhys);
        end
    end
end