function [Res, detail] = FPR_ratio(Preds, idxs_sensitive, Truelabel, num_group)
    
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
        if sum((preds) .* (1-truelabel))==0
            res = [res 1];
        else
            if sum(1-truelabel) > 0
                res = [res sum((preds) .* (1-truelabel)) / sum(1-truelabel)];
            else
                if sum((preds) .* (1-truelabel))==0 && sum(1-truelabel)==0
                    res = [res 1];
                end
            end
        end
        detail = [detail, sum((preds) .* (1-truelabel)) , sum(1-truelabel)];
    end
    
    Res = get_ave_metrics(res, 2);
end

