function Res = get_ave_metrics(metrica_mat, flag)
    
if flag == 1
    num_group = size(metrica_mat,2);
    res = zeros(size(metrica_mat,1),1);
    count = 0;
    for i = 1:(num_group-1)
        for j = (i+1) : num_group
            count = count + 1;
            res(count) = abs(metrica_mat(:,i) - metrica_mat(:,j));
        end
    end
    Res = 0.5 * (mean(res) + max(res));
else
    num_group = size(metrica_mat,2);
    res = zeros(size(metrica_mat,1),1);
    count = 0;
    for i = 1:(num_group-1)
        for j = (i+1) : num_group
            count = count + 1;
            if metrica_mat(:,i) == 0 && metrica_mat(:,j) == 0
                res(count) = 1;
            elseif metrica_mat(:,i) == 0 &&  metrica_mat(:,j) ~= 0
                res(count) = 0;
            elseif  metrica_mat(:,i) ~= 0 && metrica_mat(:,j) == 0
                res(count) = 0;
            else
                res(count) = min(metrica_mat(:,i)/metrica_mat(:,j), metrica_mat(:,j)/metrica_mat(:,i));
            end
        end
    end
    Res = 0.5 * (1 - mean(res)+ 1 - min(res));
end
if isempty(Res)
    Res = 0;
end
end