# Add `~/bin` to the `$PATH`
export PATH="$HOME/bin:$PATH";

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

# Be nice to sysadmins
if [ -f /etc/bashrc ]; then
  source /etc/bashrc
elif [ -f /etc/bash.bashrc ]; then
  source /etc/bash.bashrc
fi

# Load the dotfiles
for file in ~/bash.{path,funtions,env,alias}; do
	[ -r "$file" ] && [ -f "$file" ] && source "$file";
done;
unset file;

# enable programmable completion features
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi