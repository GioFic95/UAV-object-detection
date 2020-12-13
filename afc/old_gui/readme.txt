ISTRUZIONI

ATTENZIONE: ricordarsi di cambiare la fascia oraria del computer a quella del luogo della competizione, manualmente oppure col comando:
ln -sf /usr/share/zoneinfo/America/New_York /etc/localtime

Copiare le immagine catturate dalla telecamera in una cartella sul computer, poi avviare l'interfaccia riferendosi a quella cartella.
Se le immagini sono troppe potrebbe essere utile dare una prima scremata manualmente selezionando solo quelle relative al volo, togliendo quelle di decollo e atterraggio ad esempio.
In ogni caso per velocizzare il processo sono state implementate delle scorcatoie da tastiera (visibili nell' help integrato nell'interfaccia).

Prima di installare, aggiornare Ubuntu (o Linux) con i comandi "sudo apt-get update" e "sudo apt-get upgrade".
L'interfaccia richiede di avere installato python3 e python3-pip. Sono necessari inoltre i pacchetti opencv, PyQt5, numpy, imutils, requests.
Questi possono essere installati tramite pip (installare pip con "sudo apt install python3-pip", poi installare i pacchetti con "pip install (...)" (pip potrebbe chiamarsi invece pip3)
oppure eseguendo uno script che ho creato che installa tutto da solo (eseguendo "sudo chmod +x install_requirements_easy.sh", e poi "./install_requirements_easy.sh")
Se opencv e pyqt5 danno problemi, rimuoverli con pip, e installarli attraverso "sudo apt install python-pyqt5" e per opencv seguendo la guida (https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/).

Per fare partire l'interfaccia:
python3 user_interface.py  -dir <images_directory>  -user <username>  -pass <password>  -mission <mission_number>  -url <url>
Esempio:
python3 user_interface.py -dir ~/Desktop/images_folder  -user testuser  -pass testpass  -mission 1  -url 127.0.0.1:8000

Dopo dir va scritto il percorso della cartella contenente le immagini da processare (es:   home/cartella   , oppure solo    cartella    se il terminale già si trova in home)
Dopo "-user" va scritta l'username della squadra fornito dai giudici
Dopo "-pass" va scritta la password della squadra fornita dai giudici
Dopo ""-mission" va scritto il numero della missione assegnata dai giudici
Dopo ""-url" va scritto l'url del server dei giudici

--------------
La gara AUVSI-SUAS prevede che i partecipanti installino Docker sui loro pc, e lo utilizzino per richiedere i dati della missione, inviare la telemetria,
ricevere la posizione degli ostacoli, inviare gli oggetti rilevati con object detection etc.. Se non si volesse usare Docker in alternativa è possibile usare
le API fornite e implementare un proprio sistema (per maggiori informazioni vedere la guida dell'interoperability su Github). Questa interfaccia scritta in Python
utilizza proprio le API per comunicare con l'interperability, senza la necessità di usare l'immagine di docker "interop-client" fornita dalla competizione.

Per testare l'interfaccia, è necessario simulare (sul proprio computer o su un altro nella stessa rete) il computer dei giudici a cui inviare telemetria, oggetti etc.
Per farlo è necessario installare Docker sul proprio computer o su un altro nella stessa rete, e poi eseguire "interop-server", con i comandi riportati più avanti.
L'url del server simulato dei giudici sarà "127.0.0.1:8000" nel caso si utilizzi lo stesso computer per eseguire sia l'interfaccia che il server,
altrimenti sarà l'ip interno del computer in rete seguito da ":8000".
Prima di cominciare a testare l'interfaccia, è necessario andare andare all'url "127.0.0.1:8000" sul computer dove è in esecuzione interop-server,
accedere con "testadmin" e "testpass" e impostare i vari dettagli della missione. Invece username e password di test della squadra sono "testuser" e "testpass".

Dall'interfaccia è possibile richiedere anche i dettagli della missione (coordinate degli ostacoli, altezza minima e massima, confini dell'area di volo etc), da trasferire manualmente nel software di volo automatico.

Per ulteriori informazioni sull'utilizzo consultare l'help integrato nell'interfaccia.

--------------------------------
DOCKER
Docker praticamente è una sorta di Virtualbox, cioè permette di avere una sorta di macchina virtuale basata su ubuntu sul proprio pc (un ubuntu dentro a un altro ubuntu in pratica, ma molto più efficiente che usando VirtualBox).
Il senso di usare Docker è questo: gli sviluppatori della gara preparano un sistema operativo basato su Ubuntu, con tutti i
pacchetti necessari per la gara già installati, così che l'utente utilizzi questa "immagine" già pronta per scambiare informazioni coi giudici, senza dover installare nulla.

Docker è il programma che si occupa di gestire questi "ambienti virtuali". In pratica si apre Docker, si scarica un'"immagine" (che
contiene tutto quel sistema operativo pronto di cui si parlava), e poi la si "travasa" in un "contenitore" utilizzabile.
Una volta che l'immagine è copiata nel contenitore è possibile utilizzare il sistema operativo (tutto da terminale, non c'è un vero e proprio Desktop), aggiungere i propri codici,
installare tutto quello che si vuole, e potenzialmente anche creare una nuova immagine che contiene tutte le modifiche fatte (docker commit).
Per ulteriori informazioni consultare la guida di Docker.

-----------------------

Installazione docker:


disinstallare vecchie versioni di docker:
sudo apt-get remove docker docker-engine docker.io containerd runc


curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -


sudo add-apt-repository    "deb [arch=amd64] https://download.docker.com/linux/ubuntu  $(lsb_release -cs)  stable"


sudo apt-get update


sudo apt-get install docker-ce docker-ce-cli containerd.io

------------------------

Comandi utili docker:

verifica che docker funzioni:
sudo docker run hello-world

elenecare contenitori attivi:
sudo docker ps -a

avviare e stoppare contenitore:
sudo docker start <nome contenitore>
sudo docker stop <nome contenitore>

Avviare bash interno al contenitore (il terminale del sistema operativo) in un contenitore esistente e attivo:
sudo docker exec -it (nome_contenitore) bash
esempio:
sudo docker exec -it interop-client bash

eliminare contenitore (prima stopparlo):
sudo docker rm <nome contenitore>

-------------------------

Installazione Interoperability client o server:

Creare e avviare nuovo docker client:
sudo docker run --net=host --interactive --tty -e DISPLAY=$DISPLAY -v $HOME/Desktop/ascensore:/interop/client/ascensore -v /tmp/.X11-unix:/tmp/.X11-unix --name interop-client auvsisuas/interop-client


Creare e avviare nuovo docker server:
sudo docker run -d --restart=unless-stopped --interactive --tty     --publish 8000:80     --name interop-server     auvsisuas/interop-server

Aggiornare immagine interoperability con l'ultima rilasciata (questo cancellerà tutti i dati e le cose installate nel vecchio interop-client):

sudo docker stop interop-server
sudo docker rm interop-server
sudo docker pull auvsisuas/interop-server:latest


---------------------

altri comandi:

Comandi per elencare immagini e contenitori:
sudo docker image ls
sudo docker container ls
sudo docker system ls
sudo docker network ls

cancellare immagini e contenitori:
sudo docker image prune
sudo docker container prune
sudo docker volume prune
sudo docker network prune

Remove all stopped containers:
docker container ls -a --filter status=exited --filter status=created

Cancellare immagine:
sudo docker rmi Image Image

The cp command can be used to copy files. One specific file can be copied like:
docker cp foo.txt mycontainer:/foo.txt
docker cp mycontainer:/foo.txt foo.txt