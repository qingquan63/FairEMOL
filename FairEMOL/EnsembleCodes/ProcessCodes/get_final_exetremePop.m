function [bestPop_objs, non_idxs, non_pop, all_num] = get_final_exetremePop(base_path, obj_idx, train_valid_test, gen)

    switch train_valid_test
        case 1 
            file_names = [base_path 'detect/popobj_train%d.txt'];
        case 2
            file_names = [base_path 'detect/popobj_valid%d.txt'];
        case 3
            file_names = [base_path 'detect/popobj_ensemble%d.txt'];
        case 4
            file_names = [base_path 'detect/popobj_test%d.txt'];
    end
   
    pop = load(sprintf(file_names, gen));
    pop = pop(:,obj_idx);
    all_num = size(pop,1);
    nd = NDSort(pop,1);
    non_idxs = find(nd==1);
    non_pop = pop(non_idxs,:);
    [~, idxs] = min(non_pop,[],1);
    best_idxs = non_idxs(idxs);
    bestPop_objs = pop(best_idxs,:);
    
        

  

    

end