function select_preds = get_pop_pred(base_path, select_idxs, train_valid_test, gen)

    switch train_valid_test
        case 1 
            file_names = [base_path 'detect/pop_logits_train%d.txt'];
        case 2
            file_names = [base_path 'detect/pop_logits_valid%d.txt'];
        case 3
            file_names = [base_path 'detect/pop_logits_ensemble%d.txt'];
        case 4
            file_names = [base_path  'detect/pop_logits_test%d.txt'];
    end
   
    preds = load(sprintf(file_names, gen));
    select_preds = preds(select_idxs,:);
end