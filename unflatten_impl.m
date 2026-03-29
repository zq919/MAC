function [out, k] = unflatten_impl(flat, tmpl, k)
    % case 1: scalar AVariable
    if isa(tmpl,'AVariable') && isscalar(tmpl)
        assert(k <= numel(flat), "aux_unflatten: flat exhausted early.");
        out = flat{k};
        k = k + 1;
        return;
    end

    % case 2: AVariable object array (e.g. [u v w] or [u v; v u])
    if isa(tmpl,'AVariable') && ~isscalar(tmpl)
        n = numel(tmpl);
        assert(k + n - 1 <= numel(flat), "aux_unflatten: flat exhausted early.");

        % 预分配：用第一个元素占位（之后全覆盖）
        out = repmat(flat{k}, size(tmpl));

        % 关键：用 builtin('subsasgn') 做线性索引赋值，绕过你类里的 subsasgn 限制
        for t = 1:n
            S = substruct('()', {t});  % 线性索引
            out = builtin('subsasgn', out, S, flat{k+t-1});
        end

        k = k + n;
        return;
    end

    % case 3: cell container (possibly nested)
    if iscell(tmpl)
        out = cell(size(tmpl));
        for i = 1:numel(tmpl)
            [out{i}, k] = unflatten_impl(flat, tmpl{i}, k);
        end
        return;
    end

    error("aux_unflatten: unsupported template type.");
end