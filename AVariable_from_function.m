function avar = AVariable_from_function(var, pnts)
% 对应Julia desolve_pressure ------line 23 AVariable
% Convert a function handle + grid point arrays into an AVariable.
% pnts: cell array {X} or {X,Y} or {X,Y,Z}
% var : function handle, elementwise scalar function

    assert(iscell(pnts) && numel(pnts) >= 1, "pnts must be a non-empty cell array.");
    numDims = numel(pnts);

    % basic dimension consistency checks
    sz1 = size(pnts{1});
    for k = 2:numDims
        assert(isequal(size(pnts{k}), sz1), "All pnts arrays must have the same size.");
    end

    % evaluate mdat = var.(...) (Julia broadcast equivalent)
    switch numDims
        case 1
            X = pnts{1};
            mdat = apply_elemwise(var, X);
        case 2
            X = pnts{1}; Y = pnts{2};
            mdat = apply_elemwise(var, X, Y);
        case 3
            X = pnts{1}; Y = pnts{2}; Z = pnts{3};
            mdat = apply_elemwise(var, X, Y, Z);
        otherwise
            error("TODO: constructor for 4D or higher.");
    end

    % ensure output has the same number of dimensions as the grid
    % (MATLAB: ndims rules differ for vectors; we mainly ensure size match)
    assert(isequal(size(mdat), sz1), "Output mdat must have the same size as input grids.");

    numElems = numel(mdat);
    jac = speye(numElems);  % sparse identity, like spdiagm(ones(numElems))

    % If you have class AVariable (handle class), use it:
    % avar = AVariable(mdat, jac);
    %
    % Otherwise, return a struct:
    try
        avar = AVariable(mdat, jac);
        fprintf("AVariable is OK \n")
    catch
        avar = struct("mdata", mdat, "jac", jac);
    end
end