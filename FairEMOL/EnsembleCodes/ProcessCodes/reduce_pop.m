function [final_pop,Choose] = reduce_pop(allpop, num, plan)
allpop_temp = allpop;
Min = min(allpop,[],1);
Max = max(allpop,[],1);
Flag_noneq = Min ~= Max;
allpop(:,Flag_noneq) = (allpop(:,Flag_noneq)-min(allpop(:,Flag_noneq),[],1))./(max(allpop(:,Flag_noneq),[],1)-min(allpop(:,Flag_noneq),[],1));

non_flag = NDSort(allpop,1);
non_flag = non_flag == non_flag;

if sum(non_flag) <= num

%     if length(non_flag) > 1
%         [non_flag, max_front] = NDSort(allpop,num);
%         select_idx = find(non_flag == max_front);
%         idx = randperm(length(select_idx));
%         select_idx = select_idx(idx(1:(num - sum(non_flag < max_front))));
%         temp = (non_flag < max_front);
%         temp(select_idx) = true;
%         final_pop = allpop_temp(temp,:);
%         Choose = temp;
%     else
    final_pop = allpop_temp;
    Choose = true(1, size(final_pop,1));
%     end
else
    %%
    if plan == 1
        Del = ~non_flag;
        Remain   = find(~Del);
        Distance = pdist2(allpop,allpop);
        [~, extreme1] = max(allpop(Remain,:),[],1);
        [~, extreme2] = min(allpop(Remain,:),[],1);
        Distance(logical(eye(length(Distance)))) = inf;
        Distance(Remain(extreme1),:) = inf;
        Distance(Remain(extreme2),:) = inf;

        while sum(~Del) > num
            Remain   = find(~Del);
            Temp     = sort(Distance(Remain,Remain),2);
            [~,Rank] = sortrows(Temp);
            Del(Remain(Rank(1))) = true;
        end
        final_pop = allpop_temp(~Del,:);
    else
    %%  Two Arch
        p = 1/size(allpop,2);
        Del = ~non_flag;
        Remain   = find(~Del);
%         Distance = pdist2(allpop,allpop);
        N = size(allpop,1);
        Distance = inf(N);
        for i = 1 : N-1
            for j = i+1 : N
                Distance(i,j) = norm(allpop(i,:)-allpop(j,:),p);
                Distance(j,i) = Distance(i,j);
            end
        end
        [~, extreme1] = max(allpop(Remain,:),[],1);
        [~, extreme2] = min(allpop(Remain,:),[],1);
    %     Distance(logical(eye(length(Distance)))) = inf;
        extreme = unique([extreme1,extreme2]);
        Distance(extreme,:) = 0;
        Distance(Del,:) = 0;
        Choose = false(1, size(Del,2));
        Choose(extreme) = true;
        while sum(Choose) < num
            Remain = find(~Choose);
            [~,x]  = max(min(Distance(~Choose,Choose),[],2));
            Choose(Remain(x)) = true;
        end
        final_pop = allpop_temp(Choose,:);
    end
end
Choose = find(Choose);
end