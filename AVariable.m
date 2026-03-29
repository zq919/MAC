classdef AVariable < handle
    properties
        mdata   % double array
        jac     % sparse
    end
    methods
        function obj = AVariable(mdata, jac)
            obj.mdata = mdata;
            obj.jac = jac;
        end

                % ==========================================================
%         % Julia: function Base.:+(u::AVariable, v::AVariable)
%         % ==========================================================
%         function r = plus(u, v)
%             assert(isequal(size(u.mdata), size(v.mdata)), ...
%                 "size(u.mdata) must equal size(v.mdata).");
%             assert(isequal(size(u.jac), size(v.jac)), ...
%                 "size(u.jac) must equal size(v.jac).");
%             r = AVariable(u.mdata + v.mdata, u.jac + v.jac);
%         end

        % ==========================================================
        % Julia: Base.setindex!(res::AVariable, avar::AVariable, i, j)
        % Enables MATLAB syntax:  res(i,j) = avar
        % ==========================================================
        function obj = subsasgn(obj, S, avar)
            % Generalized setindex! for AVariable:
            % supports obj(ii,jj) = avar where ii/jj can be scalar, vector, range, or ':'
            % and synchronizes Jacobian rows.
            
                % Only handle () indexing as the first operation
                % 在 AVariable.m 的 subsasgn 里最前面加：
%                 if ~isscalar(obj)
%                     obj = builtin('subsasgn', obj, S, rhs);
%                     return;
%                 end
                % 如果不是 2 个下标（例如 obj(t)=... 或 obj(:)=...），直接交给 builtin
%                 if numel(S)>=1 && strcmp(S(1).type,'()') && numel(S(1).subs) ~= 2
%                     obj = builtin('subsasgn', obj, S, avar);
%                     return;
%                 end
                if isempty(S) || ~strcmp(S(1).type,'()')
                    obj = builtin('subsasgn', obj, S, avar);
                    return;
                end
            
                % ✅关键：不是二维赋值 obj(ii,jj)=... 的，一律交给 MATLAB
                if numel(S(1).subs) ~= 2
                    obj = builtin('subsasgn', obj, S, avar);
                    return;
                end
            
                % ✅关键：对象数组赋值也交给 MATLAB
                if ~isscalar(obj)
                    obj = builtin('subsasgn', obj, S, avar);
                    return;
                end

                if numel(S) >= 1 && strcmp(S(1).type,'()')
                    assert(numel(S(1).subs)==2, "Only 2D indexing obj(ii,jj)=... is supported.");
                    assert(ndims(obj.mdata)==2, "res.mdata must be 2D.");
            
                    ii = S(1).subs{1};
                    jj = S(1).subs{2};
            
                    [mx,my] = size(obj.mdata);
            
                    % allow ':' to mean full range
                    if ischar(ii) && strcmp(ii, ':'); ii = 1:mx; end
                    if ischar(jj) && strcmp(jj, ':'); jj = 1:my; end
            
                    assert(isnumeric(ii) && isnumeric(jj), "Indices must be numeric or ':'.");
            
                    ii = ii(:);   % column vector
                    jj = jj(:);   % column vector
            
                    % bounds check
                    assert(all(ii>=1 & ii<=mx), "Row indices out of range.");
                    assert(all(jj>=1 & jj<=my), "Col indices out of range.");
            
                    % RHS must be AVariable
                    assert(isa(avar,'AVariable'), "Right-hand side must be an AVariable.");
            
                    % --- compute global linear indices (column-major, same as Julia/MATLAB) ---
                    % Julia:
                    %   ii_glb = collect(ii) .+ ((collect(jj).-1)*mx)' ; ii_glb = ii_glb[:]
                    ii_glb_mat = ii + (jj.' - 1) * mx;   % (#ii)-by-(#jj)
                    ii_glb = ii_glb_mat(:);              % flatten column-major
            
                    % --- consistency assert (NaN-safe) ---
                    % Julia: @assert res.mdata[ii_glb]==res.mdata[ii,jj][:]
                    lhs = obj.mdata(ii_glb);        % vector
                    rhs = obj.mdata(ii, jj);        % matrix
                    assert(isequaln(lhs(:), rhs(:)), "Linear and (ii,jj) indexing mismatch.");
            
                    % --- size checks for assignment ---
                    assert(isequal(size(avar.mdata), [numel(ii), numel(jj)]), ...
                        "avar.mdata size must match the target slice size.");
                    assert(size(avar.jac,1) == numel(ii_glb), ...
                        "avar.jac must have as many rows as selected entries (numel(ii)*numel(jj)).");
                    assert(size(avar.jac,2) == size(obj.jac,2), ...
                        "avar.jac column count must match res.jac column count.");
            
                    % --- assign data ---
                    obj.mdata(ii, jj) = avar.mdata;
            
                    % --- assign Jacobian rows (may introduce extra nonzeros; same as Julia comment) ---
                    obj.jac(ii_glb, :) = avar.jac;
            
                    % apply further chained subscripts if any
                    if numel(S) > 1
                        obj = builtin('subsasgn', obj, S(2:end), avar);
                    end
                    return;
                end
            
                % fallback
                obj = builtin('subsasgn', obj, S, avar);
            end

%         function obj = subsasgn(obj, S, avar)
%             % Only handle res(i,j) = avar
%             if numel(S)==1 && strcmp(S.type,'()')
%                 assert(numel(S.subs)==2, "Only 2D indexing res(i,j)=... is supported.");
% 
%                 i = S.subs{1};
%                 j = S.subs{2};
% 
%                 numDims = ndims(obj.mdata);
%                 assert(numDims == 2, "res.mdata must be 2D.");
% 
%                 [mx,my] = size(obj.mdata);
%                 assert(i>=1 && i<=mx && j>=1 && j<=my, "Index (i,j) out of range.");
% 
%                 % column-major linear index (same as Julia/Matlab)
%                 i_glb = i + (j-1)*mx;
% 
%                 % Julia note: NaN==NaN is false. In MATLAB use isequaln.
%                 assert(isequaln(obj.mdata(i,j), obj.mdata(i_glb)), ...
%                     "res.mdata(i_glb) must equal res.mdata(i,j) under column-major indexing.");
% 
%                 % RHS must be AVariable
%                 assert(isa(avar,'AVariable'), "Right-hand side must be an AVariable.");
% 
%                 % Julia uses res.mdata[i:i,j:j] = avar.mdata[:,:]
%                 % That is a single cell assignment; we enforce scalar mdata here
%                 assert(numel(avar.mdata)==1, "This setter expects avar.mdata to be scalar (1 element).");
% 
%                 obj.mdata(i,j) = avar.mdata(:,:);
% 
%                 % Julia: res.jac[i_glb:i_glb,:] = avar.jac[:,:]
%                 obj.jac(i_glb, :) = avar.jac(:, :);
% 
%                 return;
%             end
% 
%             % Other assignments fall back to builtin behavior
%             obj = builtin('subsasgn', obj, S, avar);
%         end

        function out = subsref(obj, S)
            % Generalized getindex for AVariable:
            % - supports obj(i, jj), obj(ii, j), obj(ii, jj) where ii/jj are vectors or ranges
            % - returns AVariable(mdat_sub, jac_sub)
            
                % Handle only () indexing as the first operation
                % 在 AVariable.m 的 subsref 里最前面加：
%                 if ~isscalar(obj)
%                     out = builtin('subsref', obj, S);
%                     return;
%                 end
            % 只处理 () 索引；其它交给 builtin
                if isempty(S) || ~strcmp(S(1).type,'()')
                    out = builtin('subsref', obj, S);
                    return;
                end
            
                % ✅关键：不是二维索引 obj(ii,jj) 的，一律交给 MATLAB（支持 obj(k), obj(:) 等）
                if numel(S(1).subs) ~= 2
                    out = builtin('subsref', obj, S);
                    return;
                end
            
                % ✅关键：如果 obj 是对象数组（非标量），也交给 MATLAB 先取元素
                if ~isscalar(obj)
                    out = builtin('subsref', obj, S);
                    return;
                end

                if numel(S) >= 1 && strcmp(S(1).type, '()')
                    assert(numel(S(1).subs) == 2, "Only 2D indexing obj(ii,jj) is supported.");
            
                    assert(ndims(obj.mdata) == 2, "avar.mdata must be 2D.");
                    [mx, my] = size(obj.mdata);
            
                    ii = S(1).subs{1};
                    jj = S(1).subs{2};
            
                    % Allow ':' to mean full range
                    if ischar(ii) && strcmp(ii, ':')
                        ii = 1:mx;
                    end
                    if ischar(jj) && strcmp(jj, ':')
                        jj = 1:my;
                    end
            
                    % ii/jj must be numeric indices now
                    assert(isnumeric(ii) && isnumeric(jj), "Indices must be numeric or ':'.");
            
                    ii = ii(:);          % column
                    jj = jj(:);          % column
            
                    % bounds check (optional but safer)
                    assert(all(ii >= 1 & ii <= mx), "Row indices out of range.");
                    assert(all(jj >= 1 & jj <= my), "Col indices out of range.");
            
                    % ---- compute global linear indices (column-major) ----
                    % Julia:
                    %   ii_glb = collect(ii) .+ ((collect(jj).-1)*mx)'
                    %   ii_glb = ii_glb[:]
                    %
                    % MATLAB equivalent:
                    ii_glb_mat = ii + (jj.' - 1) * mx;  % size: (#ii)-by-(#jj)
                    ii_glb = ii_glb_mat(:);             % column-major flatten
            
                    % ---- extract sub-block (keeps 2D shape) ----
                    mdat = obj.mdata(ii, jj);           % 2D block, size (#ii)-by-(#jj)
                    jac  = obj.jac(ii_glb, :);          % rows corresponding to selected entries
            
                    % ---- consistency assert (NaN-safe) ----
                    vals_lin = obj.mdata(ii_glb);       % linear indexing gives vector
                    assert(isequaln(vals_lin(:), mdat(:)), ...
                        "Linear indexing and submatrix indexing mismatch.");
            
                    % Return a new AVariable containing the sub-data and sub-jac
                    outVar = AVariable(mdat, jac);
            
                    % If there are further chained references, apply them
                    if numel(S) > 1
                        out = builtin('subsref', outVar, S(2:end));
                    else
                        out = outVar;
                    end
                    return;
                end
            
                % Non-() access: use builtin (e.g., obj.mdata, obj.jac)
                out = builtin('subsref', obj, S);
            end
%% 标量索引
%         function out = subsref(obj, S)
%             %返回一个新的 AVariable (1x1 mdata, 1x10 jac)
%             % Only handle 2D indexing: out = obj(i,j)
%             if numel(S)==1 && strcmp(S.type,'()')
%                 assert(numel(S.subs)==2, "Only 2D indexing obj(i,j) is supported.");
%                 i = S.subs{1};
%                 j = S.subs{2};
%         
%                 numDims = ndims(obj.mdata);
%                 assert(numDims == 2, "avar.mdata must be 2D.");
%         
%                 [mx,my] = size(obj.mdata);
%                 assert(i>=1 && i<=mx && j>=1 && j<=my, "Index (i,j) out of range.");
%         
%                 % column-major linear index (same as Julia/MATLAB)
%                 i_glb = i + (j-1)*mx;
%         
%                 % Julia note: NaN==NaN is false. MATLAB: use isequaln
%                 assert(isequaln(obj.mdata(i,j), obj.mdata(i_glb)), ...
%                     "Linear index and (i,j) index mismatch.");
%         
%                 % keep 2D (1x1) slice like Julia mdata[i:i,j:j]
%                 mdat = obj.mdata(i:i, j:j);
%         
%                 % keep 2D sparse row like jac[i_glb:i_glb,:]
%                 jac  = obj.jac(i_glb:i_glb, :);
%         
%                 out = AVariable(mdat, jac);
%                 return;
%             end
%         
%             % for chained references like obj(i,j).mdata or obj.mdata(...)
%             out = builtin('subsref', obj, S);
%         end
        
        % ==========================================================
        % +  (u + v) where v is scalar OR AVariable
        % ==========================================================
        function r = plus(a, b)
            if isa(a,'AVariable') && isa(b,'AVariable')
                % AVariable + AVariable
                assert(isequal(size(a.mdata), size(b.mdata)), "mdata sizes must match.");
                assert(isequal(size(a.jac),  size(b.jac)),  "jac sizes must match.");
                r = AVariable(a.mdata + b.mdata, a.jac + b.jac);
        
            elseif isa(a,'AVariable') && isnumeric(b) && isscalar(b)
                % AVariable + scalar
                r = AVariable(a.mdata + b, a.jac * 1.0);
        
            elseif isnumeric(a) && isscalar(a) && isa(b,'AVariable')
                % scalar + AVariable
                r = AVariable(b.mdata + a, b.jac * 1.0);
        
            else
                error("Unsupported plus operands.");
            end
        end
        
        % ==========================================================
        % -  (u - v) where v is scalar OR AVariable
        % ==========================================================
        function r = minus(a, b)
            % Overload operator '-' for:
            %   AVariable - AVariable
            %   AVariable - scalar
            %   scalar   - AVariable
            %   AVariable - numeric array (same size as mdata)
            %   numeric array - AVariable (same size as mdata)
            
                if isa(a,'AVariable') && isa(b,'AVariable')
                    % AVariable - AVariable
                    assert(isequal(size(a.mdata), size(b.mdata)), "mdata sizes must match.");
                    assert(isequal(size(a.jac),  size(b.jac)),  "jac sizes must match.");
                    r = AVariable(a.mdata - b.mdata, a.jac - b.jac);
                    return;
                end
            
                if isa(a,'AVariable') && isnumeric(b)
                    if isscalar(b)
                        % AVariable - scalar
                        r = AVariable(a.mdata - b, a.jac * (+1.0));
                    else
                        % AVariable - array
                        assert(isequal(size(a.mdata), size(b)), "Array size must match u.mdata size.");
                        r = AVariable(a.mdata - b, a.jac * (+1.0));
                    end
                    return;
                end
            
                if isnumeric(a) && isa(b,'AVariable')
                    if isscalar(a)
                        % scalar - AVariable
                        r = AVariable(a - b.mdata, b.jac * (-1.0));
                    else
                        % array - AVariable
                        assert(isequal(size(a), size(b.mdata)), "Array size must match u.mdata size.");
                        r = AVariable(a - b.mdata, b.jac * (-1.0));
                    end
                    return;
                end
            
                error("Unsupported minus operands.");
            end

%         function r = minus(a, b)
%             if isa(a,'AVariable') && isa(b,'AVariable')
%                 % AVariable - AVariable
%                 assert(isequal(size(a.mdata), size(b.mdata)), "mdata sizes must match.");
%                 assert(isequal(size(a.jac),  size(b.jac)),  "jac sizes must match.");
%                 r = AVariable(a.mdata - b.mdata, a.jac - b.jac);
%         
%             elseif isa(a,'AVariable') && isnumeric(b) && isscalar(b)
%                 % AVariable - scalar
%                 r = AVariable(a.mdata - b, a.jac * (+1.0));
%         
%             elseif isnumeric(a) && isscalar(a) && isa(b,'AVariable')
%                 % scalar - AVariable
%                 r = AVariable(a - b.mdata, b.jac * (-1.0));
%         
%             else
%                 error("Unsupported minus operands.");
%             end
%         end
        
        % ==========================================================
        % *  (u * v) scalar multiplication
        % ==========================================================
        function r = mtimes(a, b)
            if isa(a,'AVariable') && isnumeric(b) && isscalar(b)
                % AVariable * scalar
                r = AVariable(a.mdata * b, a.jac * b);
        
            elseif isnumeric(a) && isscalar(a) && isa(b,'AVariable')
                % scalar * AVariable
                r = AVariable(b.mdata * a, b.jac * a);
        
            else
                error("This mtimes overload supports only scalar * AVariable or AVariable * scalar.");
            end
        end
        
        % ==========================================================
        % /  (u / v) scalar division
        % ==========================================================
        function r = mrdivide(a, b)
            if isa(a,'AVariable') && isnumeric(b) && isscalar(b)
                % AVariable / scalar
                r = AVariable(a.mdata / b, a.jac / b);
            else
                error("This mrdivide overload supports only AVariable / scalar.");
            end
        end


    end
end
