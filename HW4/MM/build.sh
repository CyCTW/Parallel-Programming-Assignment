for i in `seq 1 5`; do
	cat ~/.ssh/id_rsa.pub | ssh pp$i 'mkdir -p ~/.ssh; cat >> .ssh/authorized_keys'
done
