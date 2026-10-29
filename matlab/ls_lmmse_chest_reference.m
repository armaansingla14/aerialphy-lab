% Reference: LS vs frequency-domain LMMSE pilot channel estimation (comb-2 pilots).
% Plain MATLAB/Octave. Not executed in this repository.
nsc = 144; scs = 30e3; taus = [0 30 70 90 110 190 410]*1e-9;
pdb = [0 -1 -2 -3 -8 -17.2 -20.8]; p = 10.^(pdb/10); p = p/sum(p);
k = (0:nsc-1).'; piv = (0:2:nsc-1).';
R = zeros(nsc); for l = 1:numel(taus), R = R + p(l)*exp(-2j*pi*(k-k.')*scs*taus(l)); end
snr_db = 0:5:25; trials = 500; out = zeros(2, numel(snr_db));
for s = 1:numel(snr_db)
    nv = 10^(-snr_db(s)/10);
    W = R(:,piv+1) / (R(piv+1,piv+1) + nv*eye(numel(piv)));
    e = [0 0]; pw = 0;
    for t = 1:trials
        g = sqrt(p/2).*(randn(1,numel(p)) + 1j*randn(1,numel(p)));
        h = exp(-2j*pi*k*scs*taus) * g.';
        hp = h(piv+1) + sqrt(nv/2)*(randn(numel(piv),1) + 1j*randn(numel(piv),1));
        hls = interp1(piv, hp, k, 'linear', 'extrap');
        hmm = W*hp;
        e = e + [sum(abs(hls-h).^2) sum(abs(hmm-h).^2)]; pw = pw + sum(abs(h).^2);
    end
    out(:,s) = 10*log10(e/pw);
    fprintf('%4.0f dB  LS %.2f dB  LMMSE %.2f dB\n', snr_db(s), out(1,s), out(2,s));
end
