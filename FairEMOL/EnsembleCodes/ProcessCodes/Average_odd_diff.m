function [Res, detail] = Average_odd_diff(Preds, idxs_sensitive, Truelabel, num_group)
    
    Preds = Preds(:);
    Truelabel = double(Truelabel(:));
    Preds(Preds >= 0.5) = 1;
    Preds(Preds < 0.5) = 0;
    res = [];
    detail = [];
    for g = num_group
        gr = idxs_sensitive == g;
        truelabel = Truelabel(gr);
        preds = Preds(gr);
        if sum(truelabel) > 0 && sum(1 - truelabel) > 0
            res = [res sum((preds) .* (truelabel)) / sum(truelabel) + sum((preds) .* (1-truelabel)) / sum(1-truelabel)];
        end
        detail = [detail, sum((preds) .* (truelabel)) , sum(truelabel), sum((preds) .* (1-truelabel)), sum(1-truelabel)];
    end
    
    Res = 0.5*get_ave_metrics(res, 1);
end

