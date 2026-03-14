interface HeaderProps {
  title?: string;
}

export default function Header({ title }: HeaderProps) {
  return (
    <header className="flex h-14 items-center border-b border-gray-200 px-6 dark:border-gray-700">
      {title && (
        <h1 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
          {title}
        </h1>
      )}
    </header>
  );
}
