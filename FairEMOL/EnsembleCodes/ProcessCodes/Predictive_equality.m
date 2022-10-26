function [Res ,Detail]= Predictive_equality(Preds, idxs_sensitive, Truelabel, num_group)
    
    Preds = Preds(:);
    Truelabel = double(Truelabel(:));
    Preds(Preds >= 0.5) = 1;
    Preds(Preds < 0.5) = 0;
    res = [];
    Detail = [];
    data_ch = [];
    for g = num_group
        
        gr = idxs_sensitive == g;
        truelabel = Truelabel(gr);
        data_ch = [data_ch; [length(truelabel) - sum(truelabel), sum(truelabel)]];
        preds = Preds(gr);
        if sum(1-truelabel) > 0
            res = [res sum((preds) .* (1-truelabel)) / sum(1-truelabel)];
        end
        Detail = [Detail, [ sum((preds) .* (1-truelabel)) sum(1-truelabel)] ];
    end
    
    Res = get_ave_metrics(res, 1);
end

