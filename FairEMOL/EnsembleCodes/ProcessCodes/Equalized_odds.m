function [Res, detail] = Equalized_odds(Preds, idxs_sensitive, Truelabel, num_group)
    
    Preds = Preds(:);
    Truelabel = double(Truelabel(:));
    Preds(Preds >= 0.5) = 1;
    Preds(Preds < 0.5) = 0;
    res1 = [];
    res2 = [];
    detail = [];
    for g = num_group
        gr = idxs_sensitive == g;
        truelabel = Truelabel(gr);
        preds = Preds(gr);
        if sum(truelabel) > 0
            res1 = [res1 sum(preds .* truelabel) / sum(truelabel)];
        end
        detail = [detail, sum(preds .* truelabel),  sum(truelabel)];
        if sum(1-truelabel) > 0
            res2 = [res2 sum(preds .* (1-truelabel)) / sum(1-truelabel)];
        end
        detail = [detail, sum(preds .* (1-truelabel)),  sum(1-truelabel)];
    end
    
    Res = 0.5 * (get_ave_metrics(res1, 1) + get_ave_metrics(res2, 1));
end

