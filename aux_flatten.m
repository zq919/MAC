function flat = aux_flatten(mvars)
%AUX_FLATTEN Flatten AVariable containers to a 1D cell array.
% Accepts:
%   - AVariable scalar
%   - AVariable object array (e.g. [u v w], [u v; v u])
%   - cell arrays containing the above (can be nested)
% Returns:
%   - flat: 1D cell array {AVariable, AVariable, ...}

    flat = {};

    % ---- Case 1: AVariable scalar or array ----
    if isa(mvars, 'AVariable')
        n = numel(mvars);
        flat = cell(1, n);
        for k = 1:n
            % use builtin subsref to avoid being hijacked by overloaded subsref
            flat{k} = builtin('subsref', mvars, substruct('()', {k}));
        end
        return;
    end

    % ---- Case 2: cell container (possibly nested) ----
    if iscell(mvars)
        for k = 1:numel(mvars)
            sub = aux_flatten(mvars{k});
            flat = [flat, sub]; %#ok<AGROW>
        end
        return;
    end

    error("aux_flatten: unsupported input type. Use AVariable, AVariable array, or cell.");
end
% u = AVariable(rand(2,2), speye(4));
% v = AVariable(rand(2,2), speye(4));
% w = AVariable(rand(2,2), speye(4));
% 
% tmpl = [u v w];                 % 1x3 AVariable 对象数组
% flat = aux_flatten(tmpl);       % => 1x3 cell: {u,v,w}
% rebuilt = aux_unflatten(flat, tmpl);
% 
% disp(numel(flat));              % 3
% disp(size(rebuilt));            % [1 3]
