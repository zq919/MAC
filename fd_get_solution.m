function soln = fd_get_solution(avar, raw_soln) 
    %avar::AVariable, raw_soln::Vector{Float64}
    soln = avar.mdata + 0.0;
%     # println("soln.jac = ",avar.jac)
    soln(:) = soln(:) + avar.jac * raw_soln;
end