function [pop_ensemble, select_idxs]= select_ensemble(popobjs, select_objidx, plan)

    [FrontNo,MaxFNo] = NDSort(popobjs(:,select_objidx),zeros(size(popobjs,1),1),size(popobjs,2));
    Remains = 1:size(popobjs,1);
    Remainss = Remains(FrontNo == 1);
    popobjs = popobjs(FrontNo == 1, :);
    
    [popobjs, idx] = sortrows(popobjs,2:size(popobjs,2));
    Remainss = Remainss(idx); 
    if plan == 1
        % Extreme, the best
        pop = popobjs(:,select_objidx);
        [d, extreme_idx] = min(pop,[],1);
        pop_ensemble = popobjs(extreme_idx,:);
        select_idxs = extreme_idx;
        
    elseif plan == 2
        % knee points
        pop = popobjs(:,select_objidx);
        [FrontNo,MaxFNo] = NDSort(pop,zeros(size(pop,1),1),size(pop,2));
        non_idx = find(MaxFNo > 1);
        [~, max_idx] = max(pop,[],1);
        max_idx = unique([non_idx max_idx]);
        Remains = 1:size(pop,1);
        pop(max_idx,:) = [];
        Remains(max_idx) = [];
        FrontNo(max_idx) = [];
        [KneePoints,Distance,r,t] = FindKneePoints(pop,FrontNo,1,0.25,0.2,1);
        pop_ensemble = popobjs(Remains(KneePoints),:);
        select_idxs = Remains(KneePoints);
        
    elseif plan == 3
        % the second best
        pop = popobjs(:,select_objidx);
        [~, extreme_idx] = mink(pop,2, 1);
        if size(extreme_idx, 1) > 1
            pop_ensemble = popobjs(extreme_idx(2,:),:);
        else
            pop_ensemble = popobjs(extreme_idx,:);
        end
        select_idxs = extreme_idx;
        
    elseif plan == 4
        % all the Pareto models
        pop_ensemble = popobjs;
        select_idxs = 1:size(popobjs,1);
        
    elseif plan == 5
        % good diversity of Two archive2
        [pop_ensemble, select_idxs] = reduce_pop(popobjs(:,select_objidx), 50, 2);
        pop_ensemble = popobjs(select_idxs,:);
        
    elseif plan == 6
        % the best and second
        pop = popobjs(:,select_objidx);
        [~, extreme_idx] = mink(pop,2, 1);
        pop_ensemble = popobjs(extreme_idx(:),:);
        select_idxs = extreme_idx; 
        
    elseif plan == 7
        % EnsNoR, select the best, then delete repeat models, 
        pop = popobjs(:,select_objidx);
        [~, extreme_idx] = min(pop,[],1);
        pop_ensemble = popobjs(extreme_idx,:);
        select_idxs = extreme_idx;
        [~,temp] = unique(pop_ensemble,'rows');temp = sort(temp);
        pop_ensemble = pop_ensemble(temp,:);
        select_idxs = select_idxs(temp);
        
    elseif plan == 8
        % EnsNoRBest: 依次选择最好个体，如果有重复，就选择第二好，依次
        pop = popobjs(:,select_objidx);
        [a,~,rep_idx] = unique(pop,'rows'); % rep_idx为标签，标签相同说明是同一个模型
        rep_idxs = [];
        select_idxs = [];
        for id = select_objidx
            [~,min_ranks] = sort(pop(:,id));
            for min_rank = min_ranks'
                if isempty(select_idxs)
                    select_idxs = min_rank;
                    rep_idxs = rep_idx(min_rank);
                    break
                else
                    if ~ismember(rep_idx(min_rank),rep_idxs)
                        select_idxs = [select_idxs min_rank];
                        rep_idxs = [rep_idxs rep_idx(min_rank)];
                        break
                    end
                end
            end
        end 
        pop_ensemble = pop(select_idxs,:);
%%
    elseif plan == 9
        % Ens : 将目标分成三份，分别取出不同个模型
        better_idx = [0 4 7 8] + 1;
        similar = [1 2 5 6] + 1;
        worse = [3] + 1; 
        rep_times = [1 1 1];
        pop = popobjs(:,select_objidx);
        [a,~,rep_idx] = unique(pop,'rows'); % rep_idx为标签，标签相同说明是同一个模型
        rep_idxs = [];
        select_idxs = [];
        for r = 1:rep_times(1)
            for id = better_idx
                [~,min_ranks] = sort(pop(:,id));
                for min_rank = min_ranks'
                    if isempty(select_idxs)
                        select_idxs = min_rank;
                        rep_idxs = rep_idx(min_rank);
                        break
                    else
                        if ~ismember(rep_idx(min_rank),rep_idxs)
                            select_idxs = [select_idxs min_rank];
                            rep_idxs = [rep_idxs rep_idx(min_rank)];
                            break
                        end
                    end
                end
            end
        end
        
        for r = 1:rep_times(2)
            for id = similar
                [~,min_ranks] = sort(pop(:,id));
                for min_rank = min_ranks'
                    if isempty(select_idxs)
                        select_idxs = min_rank;
                        rep_idxs = rep_idx(min_rank);
                        break
                    else
                        if ~ismember(rep_idx(min_rank),rep_idxs)
                            select_idxs = [select_idxs min_rank];
                            rep_idxs = [rep_idxs rep_idx(min_rank)];
                            break
                        end
                    end
                end
            end
        end
        
        for r = 1:rep_times(3)
            for id = worse
                [~,min_ranks] = sort(pop(:,id));
                for min_rank = min_ranks'
                    if isempty(select_idxs)
                        select_idxs = min_rank;
                        rep_idxs = rep_idx(min_rank);
                        break
                    else
                        if ~ismember(rep_idx(min_rank),rep_idxs)
                            select_idxs = [select_idxs min_rank];
                            rep_idxs = [rep_idxs rep_idx(min_rank)];
                            break
                        end
                    end
                end
            end
        end
        pop_ensemble = pop(select_idxs,:);
        
    elseif plan == 10
        % Ens : 将目标分成三份，分别取出不同个模型
        better_idx = [0 4 7 8] + 1;
        similar = [1 2 5 6] + 1;
        worse = [3] + 1; 
        rep_times = [1 1 3];
        pop = popobjs(:,select_objidx);
        [a,~,rep_idx] = unique(pop,'rows'); % rep_idx为标签，标签相同说明是同一个模型
        rep_idxs = [];
        select_idxs = [];
        for r = 1:rep_times(1)
            for id = better_idx
                [~,min_ranks] = sort(pop(:,id));
                for min_rank = min_ranks'
                    if isempty(select_idxs)
                        select_idxs = min_rank;
                        rep_idxs = rep_idx(min_rank);
                        break
                    else
                        if ~ismember(rep_idx(min_rank),rep_idxs)
                            select_idxs = [select_idxs min_rank];
                            rep_idxs = [rep_idxs rep_idx(min_rank)];
                            break
                        end
                    end
                end
            end
        end
        
        for r = 1:rep_times(2)
            for id = similar
                [~,min_ranks] = sort(pop(:,id));
                for min_rank = min_ranks'
                    if isempty(select_idxs)
                        select_idxs = min_rank;
                        rep_idxs = rep_idx(min_rank);
                        break
                    else
                        if ~ismember(rep_idx(min_rank),rep_idxs)
                            select_idxs = [select_idxs min_rank];
                            rep_idxs = [rep_idxs rep_idx(min_rank)];
                            break
                        end
                    end
                end
            end
        end
        
        for r = 1:rep_times(3)
            for id = worse
                [~,min_ranks] = sort(pop(:,id));
                for min_rank = min_ranks'
                    if isempty(select_idxs)
                        select_idxs = min_rank;
                        rep_idxs = rep_idx(min_rank);
                        break
                    else
                        if ~ismember(rep_idx(min_rank),rep_idxs)
                            select_idxs = [select_idxs min_rank];
                            rep_idxs = [rep_idxs rep_idx(min_rank)];
                            break
                        end
                    end
                end
            end
        end
        pop_ensemble = pop(select_idxs,:);
        
    elseif plan == 11
        % Ens : 将目标分成三份，分别取出不同个模型
        better_idx = [0 4 7 8] + 1;
        similar = [1 2 5 6] + 1;
        worse = [3] + 1; 
        rep_times = [1 1 6];
        pop = popobjs(:,select_objidx);
        [a,~,rep_idx] = unique(pop,'rows'); % rep_idx为标签，标签相同说明是同一个模型
        rep_idxs = [];
        select_idxs = [];
        for r = 1:rep_times(1)
            for id = better_idx
                [~,min_ranks] = sort(pop(:,id));
                for min_rank = min_ranks'
                    if isempty(select_idxs)
                        select_idxs = min_rank;
                        rep_idxs = rep_idx(min_rank);
                        break
                    else
                        if ~ismember(rep_idx(min_rank),rep_idxs)
                            select_idxs = [select_idxs min_rank];
                            rep_idxs = [rep_idxs rep_idx(min_rank)];
                            break
                        end
                    end
                end
            end
        end
        
        for r = 1:rep_times(2)
            for id = similar
                [~,min_ranks] = sort(pop(:,id));
                for min_rank = min_ranks'
                    if isempty(select_idxs)
                        select_idxs = min_rank;
                        rep_idxs = rep_idx(min_rank);
                        break
                    else
                        if ~ismember(rep_idx(min_rank),rep_idxs)
                            select_idxs = [select_idxs min_rank];
                            rep_idxs = [rep_idxs rep_idx(min_rank)];
                            break
                        end
                    end
                end
            end
        end
        
        for r = 1:rep_times(3)
            for id = worse
                [~,min_ranks] = sort(pop(:,id));
                for min_rank = min_ranks'
                    if isempty(select_idxs)
                        select_idxs = min_rank;
                        rep_idxs = rep_idx(min_rank);
                        break
                    else
                        if ~ismember(rep_idx(min_rank),rep_idxs)
                            select_idxs = [select_idxs min_rank];
                            rep_idxs = [rep_idxs rep_idx(min_rank)];
                            break
                        end
                    end
                end
            end
        end
        pop_ensemble = pop(select_idxs,:);
        
    elseif plan ==12
        % EnsMinK: 将目标分成三份，分别选择每一维度第k小的个体
        better_idx = [0 4 7 8] + 1;
        similar = [1 2 5 6] + 1;
        worse = [3] + 1; 
        rep_times = [1 1 1];
        K = [1 1 3];
        pop = popobjs(:,select_objidx);
        [a,~,rep_idx] = unique(pop,'rows'); % rep_idx为标签，标签相同说明是同一个模型
        rep_idxs = [];
        select_idxs = [];
        for r = 1:rep_times(1)
            for id = better_idx
                [~,min_ranks] = sort(pop(:,id));
                min_ranks = min_ranks(K(1):end);
                for min_rank = min_ranks'
                    if isempty(select_idxs)
                        select_idxs = min_rank;
                        rep_idxs = rep_idx(min_rank);
                        break
                    else
                        if ~ismember(rep_idx(min_rank),rep_idxs)
                            select_idxs = [select_idxs min_rank];
                            rep_idxs = [rep_idxs rep_idx(min_rank)];
                            break
                        end
                    end
                end
            end
        end
        
        for r = 1:rep_times(2)
            for id = similar
                [~,min_ranks] = sort(pop(:,id));
                min_ranks = min_ranks(K(2):end);
                for min_rank = min_ranks'
                    if isempty(select_idxs)
                        select_idxs = min_rank;
                        rep_idxs = rep_idx(min_rank);
                        break
                    else
                        if ~ismember(rep_idx(min_rank),rep_idxs)
                            select_idxs = [select_idxs min_rank];
                            rep_idxs = [rep_idxs rep_idx(min_rank)];
                            break
                        end
                    end
                end
            end
        end
        
        for r = 1:rep_times(3)
            for id = worse
                [~,min_ranks] = sort(pop(:,id));
                min_ranks = min_ranks(K(3):end);
                for min_rank = min_ranks'
                    if isempty(select_idxs)
                        select_idxs = min_rank;
                        rep_idxs = rep_idx(min_rank);
                        break
                    else
                        if ~ismember(rep_idx(min_rank),rep_idxs)
                            select_idxs = [select_idxs min_rank];
                            rep_idxs = [rep_idxs rep_idx(min_rank)];
                            break
                        end
                    end
                end
            end
        end
        pop_ensemble = pop(select_idxs,:);
    
    elseif plan ==13
        % EnsMinK: 将目标分成三份，分别选择每一维度第k小的个体
        better_idx = [0 4 7 8] + 1;
        similar = [1 2 5 6] + 1;
        worse = [3] + 1; 
        rep_times = [1 1 1];
        K = [1 1 6];
        pop = popobjs(:,select_objidx);
        [a,~,rep_idx] = unique(pop,'rows'); % rep_idx为标签，标签相同说明是同一个模型
        rep_idxs = [];
        select_idxs = [];
        for r = 1:rep_times(1)
            for id = better_idx
                [~,min_ranks] = sort(pop(:,id));
                min_ranks = min_ranks(K(1):end);
                for min_rank = min_ranks'
                    if isempty(select_idxs)
                        select_idxs = min_rank;
                        rep_idxs = rep_idx(min_rank);
                        break
                    else
                        if ~ismember(rep_idx(min_rank),rep_idxs)
                            select_idxs = [select_idxs min_rank];
                            rep_idxs = [rep_idxs rep_idx(min_rank)];
                            break
                        end
                    end
                end
            end
        end
        
        for r = 1:rep_times(2)
            for id = similar
                [~,min_ranks] = sort(pop(:,id));
                min_ranks = min_ranks(K(2):end);
                for min_rank = min_ranks'
                    if isempty(select_idxs)
                        select_idxs = min_rank;
                        rep_idxs = rep_idx(min_rank);
                        break
                    else
                        if ~ismember(rep_idx(min_rank),rep_idxs)
                            select_idxs = [select_idxs min_rank];
                            rep_idxs = [rep_idxs rep_idx(min_rank)];
                            break
                        end
                    end
                end
            end
        end
        
        for r = 1:rep_times(3)
            for id = worse
                [~,min_ranks] = sort(pop(:,id));
                min_ranks = min_ranks(K(3):end);
                for min_rank = min_ranks'
                    if isempty(select_idxs)
                        select_idxs = min_rank;
                        rep_idxs = rep_idx(min_rank);
                        break
                    else
                        if ~ismember(rep_idx(min_rank),rep_idxs)
                            select_idxs = [select_idxs min_rank];
                            rep_idxs = [rep_idxs rep_idx(min_rank)];
                            break
                        end
                    end
                end
            end
        end
        pop_ensemble = pop(select_idxs,:);
        
    elseif plan ==14
        % EnsMinK: 将目标分成三份，分别选择每一维度第k小的个体
        better_idx = [0 4 7 8] + 1;
        similar = [1 2 5 6] + 1;
        worse = [3] + 1; 
        rep_times = [1 1 1];
        K = [1 1 10];
        pop = popobjs(:,select_objidx);
        [a,~,rep_idx] = unique(pop,'rows'); % rep_idx为标签，标签相同说明是同一个模型
        rep_idxs = [];
        select_idxs = [];
        for r = 1:rep_times(1)
            for id = better_idx
                [~,min_ranks] = sort(pop(:,id));
                min_ranks = min_ranks(K(1):end);
                for min_rank = min_ranks'
                    if isempty(select_idxs)
                        select_idxs = min_rank;
                        rep_idxs = rep_idx(min_rank);
                        break
                    else
                        if ~ismember(rep_idx(min_rank),rep_idxs)
                            select_idxs = [select_idxs min_rank];
                            rep_idxs = [rep_idxs rep_idx(min_rank)];
                            break
                        end
                    end
                end
            end
        end
        
        for r = 1:rep_times(2)
            for id = similar
                [~,min_ranks] = sort(pop(:,id));
                min_ranks = min_ranks(K(2):end);
                for min_rank = min_ranks'
                    if isempty(select_idxs)
                        select_idxs = min_rank;
                        rep_idxs = rep_idx(min_rank);
                        break
                    else
                        if ~ismember(rep_idx(min_rank),rep_idxs)
                            select_idxs = [select_idxs min_rank];
                            rep_idxs = [rep_idxs rep_idx(min_rank)];
                            break
                        end
                    end
                end
            end
        end
        
        for r = 1:rep_times(3)
            for id = worse
                [~,min_ranks] = sort(pop(:,id));
                min_ranks = min_ranks(K(3):end);
                for min_rank = min_ranks'
                    if isempty(select_idxs)
                        select_idxs = min_rank;
                        rep_idxs = rep_idx(min_rank);
                        break
                    else
                        if ~ismember(rep_idx(min_rank),rep_idxs)
                            select_idxs = [select_idxs min_rank];
                            rep_idxs = [rep_idxs rep_idx(min_rank)];
                            break
                        end
                    end
                end
            end
        end
        pop_ensemble = pop(select_idxs,:);
    elseif plan == 15
        % good diversity of Two archive2
        [pop_ensemble, select_idxs] = reduce_pop(popobjs(:,select_objidx), 9, 2);
        pop_ensemble = popobjs(select_idxs,:);
        
    end

select_idxs = Remainss(select_idxs);
end