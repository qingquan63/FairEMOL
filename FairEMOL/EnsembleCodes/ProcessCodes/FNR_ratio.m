function [Res, detail] = FNR_ratio(Preds, idxs_sensitive, Truelabel, num_group)
    
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
        if sum((1-preds) .* (truelabel))==0
            res = [res 1];
        else
            if sum(truelabel) > 0
                res = [res sum((1-preds) .* (truelabel)) / sum(truelabel)];
            else
                if sum((1-preds) .* (truelabel))==0 &&  sum(truelabel)==0
                    res = [res 1];
                end
            end
        end
        detail = [detail,sum((1-preds) .* (truelabel)) , sum(truelabel)];
    end
    
    Res = get_ave_metrics(res, 2);
end

