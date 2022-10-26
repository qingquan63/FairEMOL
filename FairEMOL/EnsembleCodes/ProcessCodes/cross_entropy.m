function cost = cross_entropy(Preds, Truelabel)
    Preds = Preds(:);
    thre = 0.9999999;
    Preds(Preds > thre) = thre;
    Preds(Preds < (1 - thre)) = 1 - thre;
    Truelabel = double(Truelabel(:));
    m = length(Truelabel);
    cost = -1.0 / m * sum( Truelabel .* log(Preds) + (1 - Truelabel) .* log(1 - Preds)) ;

