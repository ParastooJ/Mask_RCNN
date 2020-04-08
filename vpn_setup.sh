#!/bin/bash -e

read -p "What would you like to call your connection? " CONN_NAME
if [[ -n $(nmcli -m multiline c | sed -r -n "s/^NAME: +(${CONN_NAME})$/\1/p") ]]; then
    echo "Connection named ${CONN_NAME} exists."
    echo "What now?"
    QUIT="Quit with no modifications"
    MODIFY="Attempt to patch the existing connection"
    DELETE="Delete and replace the existing connection"
    select ACTION in "${QUIT}" "${MODIFY}" "${DELETE}"; do
        case ${ACTION} in
            "${QUIT}")
                exit 0
                ;;
            "${DELETE}")
                nmcli c delete ${CONN_NAME}
                nmcli c add type vpn ifname '*' vpn-type pptp con-name ${CONN_NAME}
                break
                ;;
            "${MODIFY}")
                break
                ;;
        esac
    done
else
    nmcli c add type vpn ifname '*' vpn-type pptp con-name ${CONN_NAME}
fi

read -p "ALLENINST username (i.e. chuckn): " VPN_USERNAME
read -rsp "ALLENINST password: " VPN_PASSWORD
echo
echo 'Creating connection'

echo "Setting permissions"
nmcli c m ${CONN_NAME} connection.autoconnect no
nmcli c m ${CONN_NAME} connection.permissions "user:${USER}"

echo "Setting connection parameters"
nmcli c m ${CONN_NAME} vpn.data "domain=ALLENINST, gateway=tunnel.alleninstitute.org, user=${VPN_USERNAME}, password-flags=0, refuse-pap=yes, refuse-chap=yes, refuse-eap=yes, require-mppe=yes"
nmcli c m ${CONN_NAME} vpn.secrets "password=${VPN_PASSWORD}"

echo "Setting IP configuration"
nmcli c m ${CONN_NAME} ipv4.dns "10.71.40.80,10.128.104.10,10.128.104.11"
nmcli c m ${CONN_NAME} ipv4.dns-search "corp.alleninstitute.org"
nmcli c m ${CONN_NAME} ipv4.routes "10.128.0.0/16,10.71.0.0/16 10.71.51.1"
nmcli c m ${CONN_NAME} ipv4.ignore-auto-routes yes
nmcli c m ${CONN_NAME} ipv4.ignore-auto-dns yes
nmcli c m ${CONN_NAME} ipv4.never-default yes

nmcli c m ${CONN_NAME} ipv6.method ignore

echo "All done."
echo "You can connect with \`nmcli connection up ${CONN_NAME}\` or using the GUI connection manager."

