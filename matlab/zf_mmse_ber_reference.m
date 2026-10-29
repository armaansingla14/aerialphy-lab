% Reference: uncoded BER of ZF vs MMSE detection over i.i.d. flat Rayleigh MIMO.
% Plain MATLAB/Octave (no toolboxes). Not part of the automated Python flow and
% not executed in this repository; kept as an independent cross-check script.
nt = 4; nr = 4; nsym = 2e5; snr_db = 0:5:30;
qpsk = [1+1j, 1-1j, -1+1j, -1-1j] / sqrt(2);
ber = zeros(2, numel(snr_db));
for k = 1:numel(snr_db)
    nv = 10^(-snr_db(k)/10);
    errs = [0 0]; bits = 0;
    for n = 1:nsym/nt
        H = (randn(nr,nt) + 1j*randn(nr,nt)) / sqrt(2);
        b = randi([0 1], 2, nt);
        idx = b(1,:)*2 + b(2,:) + 1;
        x = qpsk(idx).';
        y = H*x + sqrt(nv/2)*(randn(nr,1) + 1j*randn(nr,1));
        Wzf = (H'*H) \ H';
        Wmm = (H'*H + nv*eye(nt)) \ H';
        for d = 1:2
            if d == 1, xh = Wzf*y; else, xh = Wmm*y; end
            bh = [real(xh).' < 0; imag(xh).' < 0];
            errs(d) = errs(d) + sum(bh(:) ~= b(:));
        end
        bits = bits + 2*nt;
    end
    ber(:,k) = errs / bits;
    fprintf('%5.1f dB  ZF %.3e  MMSE %.3e\n', snr_db(k), ber(1,k), ber(2,k));
end
