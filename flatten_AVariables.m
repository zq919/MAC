function flat = flatten_AVariables(x)
% Flatten x into 1D cell array of AVariable, supporting AVariable arrays and nested cells.

    if isa(x,'AVariable')
        if isscalar(x)
            flat = {x};
        else
            n = numel(x);
            flat = cell(1,n);
            for k = 1:n
                % 用 builtin 线性取元素，绕开你重载的 subsref
                flat{k} = builtin('subsref', x, substruct('()', {k}));
            end
        end
        return;
    end

    if iscell(x)
        flat = {};
        for k = 1:numel(x)
            flat = [flat, flatten_AVariables(x{k})]; %#ok<AGROW>
        end
        return;
    end

    error("fd_solve: residual must be AVariable, AVariable array (e.g. [r1 r2]), or cell container.");
end