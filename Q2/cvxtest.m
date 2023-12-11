
%最少供货商
A = zeros(24,24);
A(1,1) = 56400;
for i = 2:24
    A(i,i) = 28200;
end
num_SA = 18;
num_SB = 14;
num_SC = 18;
XLS1 = XLS
cvx_begin
%cvx solver gurobi
variable SA(24,50) ;
%expression count2;

minimize(count1(SA))

subject to
0.98 * (SA*XLS1)>= A

cvx_end

%
cvx_begin
variable
minimize(1.2*GA+1.1*GB+GC)
subject to
0.98*(GA/0.6+GB/0.66+GC/0.72)>=56400
cvx_end



