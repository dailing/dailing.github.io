---
layout: post
title:  "2019-04-05-ArchLinux pacman signature error"
date:   2019-03-28 18:10:09 +0800
categories: archLinux
---
pacman -S 安装软件包出现签名错误
```
error: jansson: signature from "Eli Schwartz <eschwartz@archlinux.org>" is marginal trust
:: File /var/cache/pacman/pkg/jansson-2.10-3-x86_64.pkg.tar.xz is corrupted (invalid or corrupted package (PGP signature)).
```

问题在于签名不在信任列表里面。

可以重新生成签名：
```
sudo pacman-key --init
sudo pacman-key --populate archlinux
sudo pacman-key --refresh-keys
```

或者直接sign
```
pacman-key --lsign-key eschwartz@archlinux.org
```

可以用pacman-key 管理pacman的各种key
```
pacman-key (pacman) 5.1.3

Usage: pacman-key [options] operation [targets]

Manage pacman's list of trusted keys

Operations:
  -a, --add                 Add the specified keys (empty for stdin)
  -d, --delete              Remove the specified keyids
  -e, --export              Export the specified or all keyids
  -f, --finger              List fingerprint for specified or all keyids
  -l, --list-keys           List the specified or all keys
  -r, --recv-keys           Fetch the specified keyids
  -u, --updatedb            Update the trustdb of pacman
  -v, --verify              Verify the file(s) specified by the signature(s)
  --edit-key                Present a menu for key management task on keyids
  --import                  Imports pubring.gpg from dir(s)
  --import-trustdb          Imports ownertrust values from trustdb.gpg in dir(s)
  --init                    Ensure the keyring is properly initialized
  --list-sigs               List keys and their signatures
  --lsign-key               Locally sign the specified keyid
  --populate                Reload the default keys from the (given) keyrings
                            in '/usr/share/pacman/keyrings'
  --refresh-keys            Update specified or all keys from a keyserver

Options:
  --config <file>           Use an alternate config file (instead of
                            '/etc/pacman.conf')
  --gpgdir <dir>            Set an alternate directory for GnuPG (instead
                            of '/etc/pacman.d/gnupg')
  --keyserver <server-url>  Specify a keyserver to use if necessary

  -h, --help                Show this help message and exit
  -V, --version             Show program version

```


参考：
* <https://bbs.archlinux.org/viewtopic.php?id=233480>
* <https://www.cnblogs.com/codeaaa/p/6844331.html>
