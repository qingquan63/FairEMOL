function [Res,Details] = Equal_opportunity(Preds, idxs_sensitive, Truelabel, num_group)
    
    Preds = Preds(:);
    Truelabel = double(Truelabel(:));
    Preds(Preds >= 0.5) = 1;
    Preds(Preds < 0.5) = 0;
    res = [];
    Details = [];
    for g = num_group
        gr = idxs_sensitive == g;
        truelabel = Truelabel(gr);
        preds = Preds(gr);
        if sum(truelabel) > 0
            res = [res sum((preds) .* (truelabel)) / sum(truelabel)];
        end
        Details = [Details, [sum((preds) .* (truelabel)) sum(truelabel)]];
    end
    
    Res = get_ave_metrics(res, 1);
end

