% 将每个run中的信息储存至对应文件夹中，名为“time_info.mat”

clc,clear
cd(fileparts(mfilename('fullpath')));
addpath(genpath(cd));
%% metric info
optimized_metrics = {'Error','Average_odd_diff','Error_diff','Discovery_ratio','Predictive_equality','FOR_diff','FOR_ratio','FNR_diff','FNR_ratio'};
nonoptimized_metrics = {'Error_ratio','Discovery_diff','FPR_ratio','Disparate_impact','Statistical_parity','Equal_opportunity','Equalized_odds','Predictive_parity', 'Individual_fairness', 'Group_fairness','Accuracy'};
all_metrics = [optimized_metrics nonoptimized_metrics];

datanames ={'student','german','compas','LSAT','default','adult','bank','dutch','diabetes', 'drug_consumption','heart_failure','IBM_employee','student_academics','student_performance','patient_treatment'};
Base_path = {'F:/Fairness_data/','D:/Fairness_data/data_3_parts/','E:/fairness/ensemble_newdata/'};
select_path = [1 2 2 1 2 2 2 1 3 3 3 3 3 3 3 3];

%%
gens = [10:10:100];
objs = 1:length(optimized_metrics);


%%
for data_idx = 1:length(datanames)
    All_ALLgens_metrics_valid = [];
    All_ALLgens_metrics_test = [];
    dataname = datanames{data_idx}
    run_paths = get_filenames([Base_path,datanames]);
    for run_path_idx = 1:length(run_paths)
        disp(run_path_idx)
        Allgens_metrics_valid = [];
        Allgens_metrics_test = [];
        Allgens_metrics_ensemble = [];
        Metric_ensemble_value_test = [];
        Metrics_mat_values_test = [];
        Metric_ensemble_value_valid = [];
        Metrics_mat_values_valid = [];
        Metrics_mat_values_ensemble = [];
        Metric_ensemble_value_ensemble = [];

        gen = gens(length(gens));
        train_valid_ensemble_test_flag = 2;
        temp_name = run_paths{run_path_idx};
        mkdir([dataname '_info/'])
        try
            date_name = [Base_path{select_path(data_idx)}, dataname, temp_name((end-24):(end-5)), '/'];
            [~, non_idxs, non_pop, popsize] = get_final_exetremePop(date_name, objs, train_valid_ensemble_test_flag, gen);   % get indexs of non-dominated "models" in validation data
        catch 
            continue
        end
        non_idxs = 1:popsize;
        select_preds_valid = get_pop_pred(date_name, non_idxs, train_valid_ensemble_test_flag, gen);                 % get non-diminated "logits" in validation data
        Select_metrics_mat_valid = calculate_metric([date_name, 'detect/'], select_preds_valid, train_valid_ensemble_test_flag, optimized_metrics, nonoptimized_metrics, all_metrics{obj_idx});      % calculate non-dominated metrics in validation data
        Allgens_metrics_valid = [Allgens_metrics_valid; [gen*ones(size(Select_metrics_mat_valid,1),1) Select_metrics_mat_valid]];   % record all metrics in validation data

        train_valid_ensemble_test_flag = 3;
        select_preds_ensemble = get_pop_pred(date_name, non_idxs, train_valid_ensemble_test_flag, gen);                 % get non-diminated logits in ensemble data
        Select_metrics_mat_ensemble = calculate_metric([date_name, 'detect/'], select_preds_ensemble, train_valid_ensemble_test_flag, optimized_metrics, nonoptimized_metrics, all_metrics{obj_idx});      % calculate non-dominated metrics in validation data
        Allgens_metrics_ensemble = [Allgens_metrics_ensemble; [gen*ones(size(Select_metrics_mat_ensemble,1),1) Select_metrics_mat_ensemble]]; % record all metrics in ensemble data

        select_idx = non_idxs;   % select all the models in validation data
        train_valid_ensemble_test_flag = 4;
        select_preds_valid = get_pop_pred(date_name,non_idxs, train_valid_ensemble_test_flag, gen);     % get logits in test data
        Metrics_mat_test = calculate_metric([date_name, 'detect/'],select_preds_valid, train_valid_ensemble_test_flag, optimized_metrics, nonoptimized_metrics, all_metrics{obj_idx});  % calculate metrics of logits in test data
        Allgens_metrics_test = [Allgens_metrics_test; [gen*ones(size(Metrics_mat_test,1),1) Metrics_mat_test]];         % record all metrics in test data

        Select_preds_valid = [];
        Select_preds_ensemble = [];
        Select_preds_test = [];

        for ensemble_plan = 1:8
            [metrics_mat_ensemble, select_idx] = select_ensemble(Select_metrics_mat_ensemble, objs, ensemble_plan);
            select_idx = non_idxs(select_idx);    % select ensemble models in ensemble data

            train_valid_ensemble_test_flag = 2;
            select_preds_valid = get_pop_pred(date_name,select_idx, train_valid_ensemble_test_flag, gen);    % select [ ensemble_plan ] of ensemble and find logits in valid
            [metrics_mat_values_valid, detail_valid] = calculate_metric([date_name, 'detect/'],select_preds_valid, train_valid_ensemble_test_flag, optimized_metrics, nonoptimized_metrics, all_metrics{obj_idx}); % models: select [ ensemble_plan ] of ensemble and calculate metrics in valid
            metric_ensemble_valid = calculate_metric([date_name, 'detect/'],mean(select_preds_valid,1), train_valid_ensemble_test_flag, optimized_metrics, nonoptimized_metrics, all_metrics{obj_idx}); % ensemble values

            train_valid_ensemble_test_flag = 3;
            select_preds_ensemble = get_pop_pred(date_name,select_idx, train_valid_ensemble_test_flag, gen);    % select [ ensemble_plan ] of valid and find logits in ensemble
            [metrics_mat_values_ensemble, detail_base_ens] = calculate_metric([date_name, 'detect/'],select_preds_ensemble, train_valid_ensemble_test_flag, optimized_metrics, nonoptimized_metrics, all_metrics{obj_idx}); % models: select [ ensemble_plan ] of ensemble and calculate metrics in ensemble
            [metric_ensemble_ensemble, detail_ens_ens] = calculate_metric([date_name, 'detect/'],mean(select_preds_ensemble,1), train_valid_ensemble_test_flag, optimized_metrics, nonoptimized_metrics, all_metrics{obj_idx}); % ensemble values
            
            
            train_valid_ensemble_test_flag = 4;
            select_preds_test = get_pop_pred(date_name,select_idx, train_valid_ensemble_test_flag, gen);    % select [ ensemble_plan ] of valid and find logits in test
            [metrics_mat_values_test, detail_base_test] = calculate_metric([date_name, 'detect/'],select_preds_test, train_valid_ensemble_test_flag, optimized_metrics, nonoptimized_metrics, all_metrics{obj_idx}); % models: select [ ensemble_plan ] of ensemble and calculate metrics in test
            [metric_ensemble_test,detail_ens_test]  = calculate_metric([date_name, 'detect/'],mean(select_preds_test,1), train_valid_ensemble_test_flag, optimized_metrics, nonoptimized_metrics, all_metrics{obj_idx}); % ensemble values
            
            % base models
            metrics_mat_values_ensemble(:,1) = metrics_mat_values_ensemble(:,end); metrics_mat_values_ensemble(:,((end-2):end)) = [];
            metrics_mat_values_test(:,1) = metrics_mat_values_test(:,end); metrics_mat_values_test(:,((end-2):end)) = [];
            
            % ens model
            metric_ensemble_ensemble(:,1) = metric_ensemble_ensemble(:,end); metric_ensemble_ensemble(:,((end-2):end)) = [];
            metric_ensemble_test(:,1) = metric_ensemble_test(:,end); metric_ensemble_test(:,((end-2):end)) = [];
            
            % summary
            All_metrics = [[ [1:size(metrics_mat_values_ensemble,1)]', metrics_mat_values_ensemble];
                [-1, metric_ensemble_ensemble];
                [ [1:size(metrics_mat_values_test,1)]', metrics_mat_values_test ];
                [-1 metric_ensemble_test]];
            
            All_details = [ [ detail_base_ens, metrics_mat_values_ensemble(:, obj_idx)];
                [ detail_ens_ens, metric_ensemble_ensemble(obj_idx)];
                [ detail_base_test, metrics_mat_values_test(:, obj_idx)];
                [detail_ens_test, metric_ensemble_test(obj_idx)] ];

            % 记录
            Metrics_mat_values_valid = [Metrics_mat_values_valid; [ones(size(metrics_mat_values_valid,1),1)*ensemble_plan, metrics_mat_values_valid]];
            Metric_ensemble_value_valid = [Metric_ensemble_value_valid; [ones(size(metric_ensemble_valid,1),1)*ensemble_plan, metric_ensemble_valid]];

            Metrics_mat_values_ensemble = [Metrics_mat_values_ensemble; [ones(size(metrics_mat_values_ensemble,1),1)*ensemble_plan, metrics_mat_values_ensemble]];
            Metric_ensemble_value_ensemble = [Metric_ensemble_value_ensemble; [ones(size(metric_ensemble_ensemble,1),1)*ensemble_plan, metric_ensemble_ensemble]];

            Metrics_mat_values_test = [Metrics_mat_values_test; [ones(size(metrics_mat_values_test,1),1)*ensemble_plan, metrics_mat_values_test]];
            Metric_ensemble_value_test = [Metric_ensemble_value_test; [ones(size(metric_ensemble_test,1),1)*ensemble_plan, metric_ensemble_test]];

            Select_preds_valid = [Select_preds_valid; [ones(size(select_preds_valid,1),1)*ensemble_plan, select_preds_valid]];
            Select_preds_ensemble = [Select_preds_ensemble; [ones(size(select_preds_ensemble,1),1)*ensemble_plan, select_preds_ensemble]];
            Select_preds_test = [Select_preds_test; [ones(size(select_preds_test,1),1)*ensemble_plan, select_preds_test]];

        end
        
        Info.Allgens_metrics_test = Allgens_metrics_test;                % 每一代中test上的 所有值
        Info.Allgens_metrics_valid = Allgens_metrics_valid;              % 每一代中valid上的 所有值
        Info.Allgens_metrics_ensemble = Allgens_metrics_ensemble;        % 每一代中ensemble上的 所有值

        Info.Metric_ensemble_value_test = Metric_ensemble_value_test;    % 最后一代中test上的 ensemble后的值
        Info.Metrics_mat_values_test = Metrics_mat_values_test;          % 最后一代中test上的 用于ensemble的那些模型

        Info.Metric_ensemble_value_ensemble = Metric_ensemble_value_ensemble;    % 最后一代中ensemble上的 ensemble后的值
        Info.Metrics_mat_values_ensemble = Metrics_mat_values_ensemble;          % 最后一代中ensemble上的 用于ensemble的那些模型

        Info.Metric_ensemble_value_valid = Metric_ensemble_value_valid;  % 最后一代中valid上的 ensemble后的值
        Info.Metrics_mat_values_valid = Metrics_mat_values_valid;        % 最后一代中valid上的 用于ensemble的那些模型

        Info.Select_preds_valid = Select_preds_valid;  
        Info.Select_preds_ensemble = Select_preds_ensemble;        
        Info.Select_preds_test = Select_preds_test;  
        
        save_name = [dataname '_info' temp_name((end-24):(end-5)),'.mat'];
        save(save_name, 'Info')

    end
    
end


function files_names = get_filenames(pathname)

    files = dir(fullfile(pathname));
    files(1:2) = [];
    files_names = {};
    for i = 1:length(files)
        files_names{i} = [pathname, '/', files(i).name '/'];
    end

end