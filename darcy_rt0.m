%% RT0 mixed FEM for 2D Darcy problem on unit square
% Solves:  u + K * grad p = 0,  div u = f  in Omega
% with no-flow boundary u·n = 0 on boundary and mean(p)=0.
%
% Unknowns:
%   - RT0 flux u on edges (one DOF per edge, normal flux)
%   - P0 pressure p per element (constant)
%
% RT0 basis on a triangle T with vertices v1,v2,v3 and area A:
%   For edge i opposite vertex vi,
%     phi_i(x) = (1/(2A)) * (x - v_i)
%   Properties (with outward normal n_i on edge i):
%     phi_i · n_j = delta_ij on edge j (constant)
%     div phi_i = 1/A   (constant on T)
%
% Integration (element matrices):
%   M_ij = ∫_T (K^{-1} phi_i · phi_j) dA
%        = (1/(4A^2)) ∫_T (x - v_i)·(x - v_j) dA
%   We approximate the integral with degree-2 exact 3-point quadrature:
%     points: midpoints of each edge, weights: A/3
%   B_i   = ∫_T (div phi_i) * q dA  for P0 q=1
%        = (1/A) * A = 1
%
% This script assembles the mixed system:
%   [ M  B^T ] [u] = [0]
%   [ B  0  ] [p]   [f]
%
% and enforces no-flow boundary (u·n=0) by fixing boundary edge DOFs.

clear; clc;

% Parameters
N = 10;                 % mesh divisions per side
K = 1.0;                % scalar permeability
f_val = 1.0;            % constant source term

% Build mesh
[coords, elems] = build_mesh_unit_square(N);

% Build edges and connectivity
[edges, elem_edges, edge_signs, boundary_edges] = build_edges(coords, elems);

num_edges = size(edges, 1);
num_elems = size(elems, 1);

% Preallocate sparse matrix entries
M = spalloc(num_edges, num_edges, 9 * num_elems);
B = spalloc(num_elems, num_edges, 3 * num_elems);
F = f_val * ones(num_elems, 1); % RHS for pressure equation

% Assemble
for e = 1:num_elems
    vidx = elems(e, :);
    v = coords(vidx, :);
    A = triangle_area(v);

    % Local RT0 basis associated with the 3 edges (opposite vertices 1..3)
    % Edge order in elem_edges is consistent with vertices.
    local_edge_ids = elem_edges(e, :);
    signs = edge_signs(e, :);

    % Quadrature points: edge midpoints
    qp = [ (v(2,:) + v(3,:)) / 2;  % midpoint of edge opposite v1
           (v(1,:) + v(3,:)) / 2;  % midpoint of edge opposite v2
           (v(1,:) + v(2,:)) / 2]; % midpoint of edge opposite v3
    w = (A / 3) * ones(3,1);

    % Evaluate basis at quadrature points
    phi = zeros(3, 2, 3); % [i, comp, q]
    for i = 1:3
        for q = 1:3
            phi(i,:,q) = (qp(q,:) - v(i,:)) / (2 * A);
        end
    end

    % Local mass matrix
    Mloc = zeros(3,3);
    for i = 1:3
        for j = 1:3
            s = 0;
            for q = 1:3
                s = s + w(q) * (phi(i,:,q) * phi(j,:,q)');
            end
            Mloc(i,j) = (1 / K) * s;
        end
    end

    % Local divergence matrix (B)
    % div(phi_i) = 1/A, integrated against constant test gives 1.
    Bloc = ones(1,3); % size 1x3 for this element

    % Scatter to global with orientation signs
    for i = 1:3
        ei = local_edge_ids(i);
        si = signs(i);
        B(e, ei) = B(e, ei) + si * Bloc(i);
        for j = 1:3
            ej = local_edge_ids(j);
            sj = signs(j);
            M(ei, ej) = M(ei, ej) + si * sj * Mloc(i,j);
        end
    end
end

% Apply no-flow boundary: fix boundary edge fluxes to zero
fixed = boundary_edges;
free = setdiff(1:num_edges, fixed);

% Build saddle-point system
% [M_ff  B_f^T][u_f] = [0]
% [B_f   0   ][p  ]   [F]
M_ff = M(free, free);
B_f = B(:, free);

% Fix pressure nullspace by setting mean pressure = 0
% Use a simple constraint: set the last element pressure to zero
p_fixed = num_elems;
free_p = 1:(num_elems-1);

% Reduced system
A_sys = [M_ff, B_f(free_p,:)';
         B_f(free_p,:), sparse(num_elems-1, num_elems-1)];

rhs = [zeros(length(free),1); F(free_p)];

sol = A_sys \ rhs;

u_free = sol(1:length(free));
p = zeros(num_elems,1);
p(free_p) = sol(length(free)+1:end);

% Insert fixed values
u = zeros(num_edges, 1);
u(free) = u_free;

% Report
fprintf('RT0 Darcy solve complete.\n');
fprintf('Number of elements: %d\n', num_elems);
fprintf('Number of edges: %d (fixed boundary: %d)\n', num_edges, length(fixed));

% Plot pressure per element
trisurf(elems, coords(:,1), coords(:,2), p, 'EdgeColor', 'k');
view(2); axis equal tight; colorbar;
title('P0 Pressure');

%% ----- helper functions -----
function [coords, elems] = build_mesh_unit_square(N)
% Uniform triangulation of [0,1]^2 into 2*N^2 triangles.
[x, y] = meshgrid(linspace(0,1,N+1), linspace(0,1,N+1));
coords = [x(:), y(:)];

% Triangulate each square into two triangles
elems = zeros(2*N*N, 3);
idx = 1;
for j = 1:N
    for i = 1:N
        n1 = (j-1)*(N+1) + i;
        n2 = n1 + 1;
        n3 = n1 + (N+1);
        n4 = n3 + 1;
        % Triangle 1 (n1, n2, n4)
        elems(idx, :) = [n1, n2, n4];
        idx = idx + 1;
        % Triangle 2 (n1, n4, n3)
        elems(idx, :) = [n1, n4, n3];
        idx = idx + 1;
    end
end
end

function [edges, elem_edges, edge_signs, boundary_edges] = build_edges(coords, elems)
% Build global edge list and orientation signs per element
num_elems = size(elems,1);
edge_map = containers.Map('KeyType','char','ValueType','int32');
edges = zeros(0,2);
elem_edges = zeros(num_elems,3);
edge_signs = zeros(num_elems,3);

for e = 1:num_elems
    v = elems(e,:);
    % local edges: opposite vertices 1..3 -> (2,3), (3,1), (1,2)
    % oriented consistently with CCW element ordering
    local = [v(2) v(3); v(3) v(1); v(1) v(2)];
    for i = 1:3
        a = local(i,1); b = local(i,2);
        if a < b
            key = sprintf('%d_%d', a, b);
            edge = [a b];
        else
            key = sprintf('%d_%d', b, a);
            edge = [b a];
        end
        if isKey(edge_map, key)
            eid = edge_map(key);
        else
            eid = size(edges,1) + 1;
            edges(eid,:) = edge;
            edge_map(key) = eid;
        end
        elem_edges(e,i) = eid;
        % sign relates local outward normal to global edge orientation
        edge_signs(e,i) = local_edge_outward_sign(coords, v, local(i,:), edge);
    end
end

% Boundary edges: those belonging to only one element
counts = zeros(size(edges,1),1);
for e = 1:num_elems
    for i = 1:3
        counts(elem_edges(e,i)) = counts(elem_edges(e,i)) + 1;
    end
end
boundary_edges = find(counts == 1);
end

function s = local_edge_outward_sign(coords, v, local_edge, global_edge)
% Determine sign between local outward normal and global edge normal.
% Assumes element vertices v are in CCW order.
pts = coords(v,:);

% Local edge (node indices in element ordering)
a = local_edge(1); b = local_edge(2);
pa = coords(a,:); pb = coords(b,:);
t_local = pb - pa;
n_out = [t_local(2), -t_local(1)]; % outward normal for CCW element

% Global edge orientation (from lower index to higher index)
ga = global_edge(1); gb = global_edge(2);
pga = coords(ga,:); pgb = coords(gb,:);
t_global = pgb - pga;
n_global = [t_global(2), -t_global(1)];

% Decide sign based on alignment of normals
s = sign(dot(n_out, n_global));
if s == 0
    s = 1;
end
end

function A = triangle_area(v)
A = 0.5 * det([v(2,:) - v(1,:); v(3,:) - v(1,:)]);
A = abs(A);
end
