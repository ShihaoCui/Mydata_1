% X = [];
X = [x00,x01, x02,xx01,xx02,xx03,xx04,xx05,xx06,xx07,...10
    x11, x13, x14, x15, x16, x17, x18,...7
    x21, x22, x23, x24, x25, x26, x27, x28,xx20,xx21,...10
    x31, x310, x311, x32, x33, x34, x35, x36, x37, x38, x39];...11

XX = reshape(X,[930,1050,3,38]);
size(XX)
save XX XX