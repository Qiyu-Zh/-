function f=count1(S)
sum_row = sum(S)
k = 0;
for i = 1:1:length(sum_row)
    m = sum_row(1,i)
    
    if m>0
        k = k+1;
    end
end
f = k; 
end

