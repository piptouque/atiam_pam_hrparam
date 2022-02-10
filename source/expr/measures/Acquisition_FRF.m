        
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% ACQUISITION
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all

%% Parametres
fs = 51200; %Fr�quence d'�chantillonnage
tps = 5; %Temps d'acquisition
Sensibilite_Marteau = 23.61e-3; %Unite A MODIF
Sensibilite_Accelero = 10.17e-3; %V/m.s^-2Unite A MODIF
Sensibilite_Micro = 67.8e-3; % V/Pa �
% � faire: sensibilit� micro

%% Acquisition Carte NI (apr�s lancement de Init)
s.Rate = fs;
s.DurationInSeconds= tps;
[data,time] = s.startForeground();
s.wait(); 
s.stop();
s.release();

Data.Time = time;
signal_marteau = data(:,1) /Sensibilite_Marteau;
signal_accelero = data(:,2)/Sensibilite_Accelero;
signal_micro = data(:, 3)/Sensibilite_Micro;

%% Fenetrage des signaux

%Fenetrage porte du signal du marteau
largeur = 80; %nbre de point
[signal_marteau_fen,Indice_deb,Fenetre_m_libre_1] = FenetragePorte(signal_marteau,largeur);

%Fenetrage exponentiel de la r�ponse de l'acc�l�rom�tre  
Taux = 1e-4;
%deb_libre_1 = find(signal_marteau == max(signal_marteau));
[signal_accelero_fen,Fenetre_libre_acc_1] = FenetrageExponentiel(signal_accelero, Taux, Indice_deb);

%Fenetrage porte du signal du microphone
largeur = 80; %nbre de point
[signal_micro_fen,Fenetre_micro_libre_1] = FenetrageExponentiel(signal_micro,Taux, Indice_deb);



%% Calcul de la FRF et de la coh�rence
Nfft = 2^nextpow2(length(signal_accelero));
freq = [0:fs/Nfft:fs/2-1/Nfft];

F = fft(signal_marteau,Nfft);
F_fenetre = fft(signal_marteau_fen,Nfft);
A = fft(signal_accelero,Nfft);
A_fenetre = fft(signal_accelero_fen,Nfft);

FRF = CalculFRF_H1(F_fenetre,A_fenetre);

%% Affichage des signaux
scrsz = get(groot,'ScreenSize');
figure('Position',[scrsz(3)/4 1 scrsz(3)/2 scrsz(4)])

axTemp1 = subplot(4,2,1); %Signaux temporels du marteau (non-fenetr�s et fenetr�s)
plot(time,signal_marteau); hold on
plot(time,signal_marteau_fen, 'r'),
plot(time,Fenetre_m_libre_1*max(signal_marteau),'r')
title('marteau')
xlabel('temps [s]')
ylabel('Force [N]')

axTemp2 = subplot(4,2,3); %Signaux fr�quentiels de l'acc�l�rom�tre(non-fenetr�s et fenetr�s)
plot(time,signal_accelero), hold on
plot(time,signal_accelero_fen, 'r')
plot(time,Fenetre_libre_acc_1*max(signal_accelero),'r')
title('accelerometre')
xlabel('temps [s]')
ylabel('Acc�l�ration [m.s^{-1}]')

axFreq1 = subplot(4,2,2); %Signaux fr�quentiels du marteau (non-fenetr�s et fenetr�s)
plot(freq,db(F(1:Nfft/2))), hold on
plot(freq,db(F_fenetre(1:Nfft/2)), 'r')
title('marteau')
xlabel('frequence [s]')
ylabel('Force [N]')

axFreq2 = subplot(4,2,4); %Signaux temporels de l'acc�l�rom�tre(non-fenetr�s et fenetr�s)
plot(freq,db(A(1:Nfft/2))), hold on
plot(freq,db(A_fenetre(1:Nfft/2)), 'r')
title('accelerometre')
xlabel('fr�quence (Hz)')
ylabel('Acc�l�ration [m.s^{-2}]')

axFreq3 = subplot(4,1,3); %Module de la FRF
plot(freq,db(FRF(1:Nfft/2)))
xlabel('fr�quence (Hz)')
ylabel('Acc�l�rance [dB]')

axFreq4 = subplot(4,1,4);  %Phase de la FRF
plot(freq,unwrap(angle(FRF(1:Nfft/2))))
xlabel('fr�quence (Hz)')
ylabel('Phase [deg]')

linkaxes([axFreq1,axFreq2,axFreq3,axFreq4],'x')
linkaxes([axTemp1,axTemp2],'x')

Final.Data = Data;
Final.marteau.fen = signal_marteau_fen;
Final.accelero.fen = signal_accelero_fen;
Final.micro.fen = signal_micro_fen;
Final.marteau.brut = signal_marteau;
Final.accelero.brut = signal_accelero;
Final.micro.brut = signal_micro;
Final.FRF = FRF(1:Nfft/2);
Final.freq = freq;
Final.time = time;
Final.fs = fs;
Final.fen.start = Indice_deb;
Final.fen.length = largeur;
file_name = 'FirstStringTwelfthFret_Finger_Bridge_TensionMod_3.mat';
Final.excitation_type = 'Finger';

save(file_name, 'Final');
