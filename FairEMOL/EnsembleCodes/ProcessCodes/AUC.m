function [res, total_AUC] = AUC(preds, idxs_sensitive, truelabel, num_group)


    for g = num_group
        gr = idxs_sensitive == g;
        res(g) = roc_curve(preds(gr), truelabel(gr),0);
    end
    
    total_AUC = roc_curve(preds, truelabel,0);
end