function out = apply_elemwise(fun, varargin)
% Try vectorized evaluation; if it fails, fall back to arrayfun (elementwise).
    try
        out = fun(varargin{:}); % vectorized version (preferred)
    catch
        out = arrayfun(fun, varargin{:}); % elementwise version
    end
end