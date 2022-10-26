function res = Accuracy(preds, idxs_sensitive, truelabel, num_group)

    preds(preds >= 0.5) = 1;
    preds(preds < 0.5) = 0;
    preds = preds(:);
    truelabel = truelabel(:);
    res = sum(preds == truelabel)/length(truelabel);
end