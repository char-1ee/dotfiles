#!/bin/bash
files="bashrc vimrc"

for file in $files; do
	ln -s ~/dotfiles/$file ~/.$file
done
