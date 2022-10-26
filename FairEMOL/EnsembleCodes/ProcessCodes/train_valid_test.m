function [sens_path, true_label] = train_valid_test(base_path, idx_idx)
    

    switch idx_idx
        case 1
            % train
            sens_path = load([base_path '\train_idxs_sensitive.txt']);
            true_label = load([base_path '\train_truelabel.txt']);
        case 2
            % valid
            sens_path = load([base_path '\valid_idxs_sensitive.txt']);
            true_label = load([base_path '\valid_truelabel.txt']);
        case 3
            % valid
            sens_path = load([base_path '\ensemble_idxs_sensitive.txt']);
            true_label = load([base_path '\ensemble_truelabel.txt']);
            
        case 4
            % test
            sens_path = load([base_path '\test_idxs_sensitive.txt']);
            true_label = load([base_path '\test_truelabel.txt']);
    end
end